# HybridLinearKVPool support for decode KV cache offload

## Goal

`DecodeKVCacheOffloadManager` currently supports decode-side offload only when
`token_to_kv_pool_allocator.get_kvcache()` returns `MHATokenToKVPool` or
`MLATokenToKVPool`. For hybrid linear/Mamba models this returns
`HybridLinearKVPool`, so initialization fails before decode-side KV offload can
be used.

The implementation should add support for `HybridLinearKVPool` without
duplicating the existing Mamba host-transfer stack that is already used by
HiCache.

## Existing building blocks

### Device pools

`HybridLinearKVPool` is a wrapper over two logical device-side stores:

- `full_kv_pool`: the full-attention cache. This is usually `MHATokenToKVPool`,
  or `MLATokenToKVPool` when `use_mla=True`.
- `mamba_pool`: the linear/Mamba state pool owned by `HybridReqToTokenPool`.

The wrapper exposes global model-layer ids and maps full-attention layers into
the compact dense layer ids used by `full_kv_pool`.

### Host pools

The host-side pieces already exist:

- `MHATokenToKVPoolHost` / `AsymmetricMHATokenToKVPoolHost` for MHA full
  attention KV.
- `MLATokenToKVPoolHost` for MLA full attention KV.
- `MambaPoolHost` for Mamba state. It supports page-first layouts and implements
  D2H/H2D transfer plus generic and zero-copy storage page metadata.
- `HostPoolGroup` for grouping an anchor KV host pool with side pools.

Functionally, `HostPoolGroup([KV, MAMBA])` is already a
`HybridLinearKVPoolHost`: it exposes the normal host-pool interface through the
KV anchor while routing side-pool transfers through `PoolTransfer`.

### Controllers

`HiCacheController` handles a single host pool. Its constructor explicitly
unwraps `HybridLinearKVPool` to `full_kv_pool`, which is correct for ordinary
KV-only paths but loses Mamba state.

`HybridCacheController` extends the same API with:

- `write(..., extra_pools=...)`
- `load(..., extra_pools=...)`
- `write_storage(..., extra_pools=...)`
- `prefetch(..., extra_pools=...)`

It also registers each host pool with v2 storage backends and handles
`PoolTransfer` allocation, index movement, D2H/H2D transfer, and storage I/O.

## Recommended design

Prefer reusing the existing hybrid HiCache abstractions instead of adding a new
independent offload stack.

For `HybridLinearKVPool`, `DecodeKVCacheOffloadManager.__init__` should:

1. Build a full-attention host pool from `kv_cache.full_kv_pool`.
2. Build a `MambaPoolHost` from `req_to_token_pool.mamba_pool`.
3. Wrap them in `HostPoolGroup` with two `PoolEntry`s:
   - `PoolName.KV` as the primary index anchor.
   - `PoolName.MAMBA` as an extra pool.
4. Use `HybridCacheController` instead of `HiCacheController`.
5. Set `transfer_layer_num` to the union of full-attention global layer ids and
   Mamba global layer ids.

This mirrors `build_hybrid_mamba_stack()` in
`python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`, but the
decode offload manager does not currently have a `CacheInitParams` object, so a
small local helper is cleaner than forcing the decode path through the full
assembler.

Pseudo-code:

```python
if isinstance(kv_cache, HybridLinearKVPool):
    full_host_pool = build_full_kv_host_pool(kv_cache.full_kv_pool)
    mamba_host_pool = MambaPoolHost(
        req_to_token_pool.mamba_pool,
        server_args.hicache_ratio,
        mamba_host_size,
        allocator_type=allocator_type,
        layout=server_args.hicache_mem_layout,
    )
    full_mapping = dict(kv_cache.full_attention_layer_id_mapping)
    mamba_mapping = dict(req_to_token_pool.mamba_map)
    transfer_layer_num = len(full_mapping | mamba_mapping)
    host_group = HostPoolGroup([
        build_pool_entry(... PoolName.KV ..., is_anchor=True),
        build_pool_entry(
            ... PoolName.MAMBA ...,
            device_alloc_fn=req_to_token_pool.mamba_allocator.alloc,
            device_free_fn=req_to_token_pool.mamba_allocator.free,
        ),
    ])
    controller = HybridCacheController(..., host_group, transfer_layer_num=transfer_layer_num)
```

If a concrete `HybridLinearKVPoolHost` class is desired for readability, make it
a thin factory/wrapper around `HostPoolGroup`, not a new implementation of
allocation or transfer logic.

## Mamba offload semantics for decode

KV cache and Mamba state do not have the same cardinality.

Decode KV offload writes page-aligned incremental token ranges:

```text
device KV indices: req_to_token[req_pool_idx][start:end]
host KV indices:   same length, allocated from the KV anchor pool
storage keys:      one key per KV page hash
```

Mamba state is request state, not per-token KV. A request normally has one
active `req.mamba_pool_idx`. With extra-buffer tracking, the cacheable state may
live in the ping-pong tracking buffer at a configured Mamba track boundary.

Therefore decode-side Mamba offload should use `PoolTransfer` with
`PoolHitPolicy.TRAILING_PAGES`:

- On each incremental KV offload, include Mamba only when there is a valid
  checkpoint state corresponding to the end of the offloaded chunk.
- The transfer should have one Mamba host slot and one Mamba device slot.
- The storage key should be the last page hash of the offloaded chunk.
- If no valid checkpoint exists for that end position, offload KV only.

Recommended helper:

```python
def _build_mamba_backup_transfer(req, chunk_end: int) -> Optional[list[PoolTransfer]]:
    mamba_device_index = _select_mamba_checkpoint_index(req, chunk_end)
    if mamba_device_index is None:
        return None
    return [
        PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=None,              # allocated by HybridCacheController.write
            device_indices=mamba_device_index.view(1),
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
    ]
```

The checkpoint selection must match the model's Mamba tracking rules:

- Without extra-buffer tracking, the active `req.mamba_pool_idx` is the only
  available state. It is safest to write this state only when the request is
  finishing, because intermediate decode state can still be mutated by later
  tokens.
- With extra-buffer tracking, use the ping-pong keep slot only when
  `req.mamba_last_track_seqlen == chunk_end`. This gives a state aligned to the
  KV prefix being archived.
- In lazy extra-buffer mode, honor `req.mamba_lazy_is_insert`; if the current
  state was marked invalid, skip the Mamba side transfer.

This mirrors the `HiMambaRadixCache` convention where Mamba state is archived as
a trailing-page side pool keyed by the final page hash of the KV prefix.

## Write and storage flow changes

`offload_kv_cache(req)` should keep the existing KV range calculation and add
hybrid side-pool metadata:

1. Compute `incremental_tokens`, `incremental_indices`, `start`, and `end` as
   today.
2. Build `extra_pools = _build_mamba_backup_transfer(req, end)` for hybrid
   caches.
3. Call:

   ```python
   host_indices = self.cache_controller.write(
       device_indices=incremental_indices.long(),
       node_id=ack_id,
       extra_pools=extra_pools,
   )
   ```

4. Store `extra_pools` in `ongoing_offload[ack_id]` so the storage backup can
   reuse the allocated Mamba host index.
5. After D2H completion, compute page hashes as today.
6. Before `write_storage`, assign the Mamba transfer key to the last page hash:

   ```python
   if extra_pools and page_hashes:
       extra_pools[0].keys = [page_hashes[-1]]
   self.cache_controller.write_storage(
       host_indices,
       incremental_tokens,
       hash_value=page_hashes,
       extra_pools=extra_pools,
   )
   ```

`HybridCacheController._page_backup()` will write both the Mamba side pool and
the anchor KV pool. For MLA hybrid models it already preserves the base
optimization: replicated MLA KV is written by TP0, while rank-sharded Mamba
state is still written by every TP rank.

## Host memory sizing

When `--hicache-size` is set, split the fixed GB budget between full KV and
Mamba pools using the same proportional logic as `_split_hicache_size()`:

```text
full_host_size  = hicache_size * full_kv_bytes / total_bytes
mamba_host_size = hicache_size * mamba_bytes   / total_bytes
```

When `--hicache-size <= 0`, pass the same `hicache_ratio` to both pools.

The Mamba host pool has `page_size=1` internally. It can still share the same
storage page hash as the KV anchor because the side-pool transfer is trailing
state, not a token-page array.

## Release and lifecycle changes

The current decode offload manager releases finished requests through:

```python
self.req_to_token_pool.free(req)
```

For `HybridReqToTokenPool`, this only frees the request row. It does not free
`req.mamba_pool_idx` or ping-pong tracking slots. Hybrid support must add an
explicit Mamba release step, equivalent to the normal `release_kv_cache()` path:

- Free `req.mamba_pool_idx` when it is still owned by the request.
- Free valid `req.mamba_ping_pong_track_buffer` slots, respecting lazy mode and
  invalid `-1` entries.
- Clear request-side Mamba fields after release.

Also release auto-allocated Mamba host slots after storage backup. With
`HybridCacheController`, extra-pool host slots are not freed by
`decode_host_mem_pool.free(host_indices)`, because the anchor KV pool and the
Mamba pool have independent allocators. `_check_backup_progress()` should call:

```python
self.cache_controller.append_host_mem_release(
    host_indices,
    extra_pools=operation.pool_transfers,
)
```

or directly free `transfer.host_indices` for `PoolName.MAMBA` after backup ack.
Use the controller helper if storage queues are enabled because it already
handles page-wise host release queues for extra pools.

## Storage backend requirements

Generic storage backends can store Mamba pages through `batch_set_v2` /
`batch_get_v2` if their base implementation is complete for registered pools.
Zero-copy backends need v2 pool registration:

```python
storage_backend.register_mem_host_pool_v2(mamba_host_pool, PoolName.MAMBA)
```

`HybridCacheController.attach_storage_backend(..., host_pools=host_group.entries)`
already performs this registration. Decode offload manager should pass
`host_pools` when attaching storage, or construct `HybridCacheController` with
the startup `storage_backend` and `mem_pool_host=HostPoolGroup`.

Backends with known Mamba handling include Mooncake, HF3FS, and NIXL paths that
look up `PoolName.MAMBA` in their v2 registration tables.

## Suggested implementation steps

1. Import hybrid pieces in `decode_kvcache_offload_manager.py`:
   - `HybridLinearKVPool`
   - `MambaPoolHost`
   - `HostPoolGroup`
   - `PoolName`, `PoolTransfer`, `PoolHitPolicy`
   - `HybridCacheController`
   - `build_pool_entry` and possibly `_split_hicache_size`, or local equivalents.
2. Add a helper that builds the full KV host pool for MHA/MLA.
3. Add a helper that builds the hybrid host group and returns
   `(host_group, transfer_layer_num)`.
4. Select `HybridCacheController` for `HybridLinearKVPool`; keep
   `HiCacheController` for existing MHA/MLA paths.
5. Extend `ongoing_offload` and `ongoing_backup` records to carry
   `extra_pools`.
6. Add `_build_mamba_backup_transfer(req, end)` and call controller
   `write(..., extra_pools=...)`.
7. Update `_trigger_backup()` to attach trailing Mamba storage keys and call
   `write_storage(..., extra_pools=...)`.
8. Update backup ack handling to free both anchor KV host indices and Mamba host
   indices.
9. Update finished-request release to free request-owned Mamba device slots.
10. Add tests for initialization, D2H write invocation, storage transfer metadata,
    and release behavior.

## Test plan

Unit tests should avoid requiring real GPU kernels where possible by using
mocked host pools/controllers:

- `DecodeKVCacheOffloadManager` initializes with a fake `HybridLinearKVPool`
  and chooses `HybridCacheController`.
- Hybrid host group contains `PoolName.KV` and `PoolName.MAMBA`.
- `offload_kv_cache()` passes `PoolTransfer(name=PoolName.MAMBA)` only when a
  valid Mamba checkpoint exists.
- `_trigger_backup()` sets the Mamba transfer key to the last page hash.
- `_check_backup_progress()` frees both KV host slots and Mamba host slots.
- Finished request release frees `req.mamba_pool_idx` and ping-pong slots.

Integration tests should cover:

- Hybrid MHA + Mamba with `hicache_mem_layout=page_first`.
- Hybrid MLA + Mamba, verifying nonzero TP ranks still back up Mamba side data.
- Lazy extra-buffer mode, verifying invalid Mamba states are not archived.
- A v2 storage backend path such as Mooncake or NIXL if available in CI.

## Main risks

- Archiving a Mamba state whose sequence length does not match the KV prefix
  hash. This would restore a semantically wrong state even if the storage hit is
  successful.
- Leaking Mamba device slots because decode offload currently bypasses
  `release_kv_cache()`.
- Leaking Mamba host slots because `HostPoolGroup.free()` frees only the anchor
  pool.
- Accidentally using `HiCacheController` with `HybridLinearKVPool`, which drops
  side-pool transfers by unwrapping to `full_kv_pool`.
- Storage backend mismatch when using zero-copy without registering
  `PoolName.MAMBA`.
