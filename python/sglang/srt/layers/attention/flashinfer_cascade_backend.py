"""FlashInfer Cascade attention backend.

Subclasses ``FlashInferAttnBackend`` and adds a
``MultiLevelCascadeAttentionWrapper`` decode path that batches the leading
shared-prefix portion of the running batch into a single matmul. The stock
``flashinfer`` backend runs ``BatchDecodeWithPagedKVCacheWrapper`` per-request
even when prefixes are deduped at the storage layer (RadixAttention shares
pages, but Q.K is still computed per-request). FlashInfer ships
``MultiLevelCascadeAttentionWrapper`` which can batch the shared portion
across requests; this backend wires it into SGLang's decode path.

Scope:
    * Decode-only. Extend (prefill) falls through to the parent class.
    * Eager mode: cascade fires when (1) the detected shared-prefix length
      passes ``--cascade-min-prefix-tokens`` and (2) batch size passes
      ``--cascade-min-batch-size``. Otherwise the parent per-request path
      runs, unless ``SGLANG_CASCADE_FORCE_NO_PREFIX=1`` is set.
    * CUDA-graph mode: every captured ``cuda_graph_bs`` gets its own
      wrapper plus pre-allocated indptr/indices buffers; ``plan()`` writes
      into those buffers per replay step (host-side, before the graph
      fires); the captured ``run()`` reads from the same addresses.
      Cascade is always armed in CG-mode regardless of threshold:
      ``common_prefix=0`` is mathematically equivalent to per-request
      decode plus a no-op level-0 launch, so correctness holds at every
      bs in the captured list while keeping the capture graph singular.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import torch

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.utils import is_flashinfer_available

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

if is_flashinfer_available():
    from flashinfer import MultiLevelCascadeAttentionWrapper


class _CascadePlanState(msgspec.Struct):
    """Per-step cascade plan state. Set by ``init_forward_metadata`` (eager)
    or ``init_forward_metadata_out_graph`` (CG) when the cascade arms;
    consumed by ``forward_decode``. ``None`` means cascade did not fire for
    this step (parent's per-request path runs instead).
    """

    common_prefix_tokens: int
    bs: int
    graph_key: Optional[Any] = None
    perm: Optional[torch.Tensor] = None
    inv_perm: Optional[torch.Tensor] = None


class FlashInferCascadeAttnBackend(FlashInferAttnBackend):
    """FlashInfer backend with a cross-request shared-prefix decode path.

    Detection (eager): walks each request's leading slot indices in
    ``req_to_token``, finds the longest run where every request's slot
    matches request 0's. Cascade fires when that run is at least
    ``cascade_min_prefix_tokens`` long AND batch size is at least
    ``cascade_min_batch_size``. Set ``SGLANG_CASCADE_FORCE_NO_PREFIX=1`` to
    keep using cascade when the batch has no detected prefix sharing.

    Detection (CG-mode): same algorithm, but driven from ``req_pool_indices``
    + ``seq_lens`` (forward_batch is not available at replay-time).

    Under CUDA graphs every captured ``cuda_graph_bs`` gets its own wrapper
    plus pre-allocated indptr/indices buffers; ``plan()`` writes into those
    buffers per replay step (host-side, before the graph fires); the captured
    ``run()`` reads from the same addresses. Cascade is always armed in
    CG-mode regardless of threshold: cascade with ``common_prefix=0`` is
    mathematically equivalent to per-request decode plus a no-op level-0
    launch, so correctness holds at every bs in the captured list while
    keeping the capture graph singular (instead of capturing two arms per
    bs).
    """

    def __init__(self, model_runner: ModelRunner, **kwargs):
        super().__init__(model_runner, **kwargs)

        self.cascade_min_prefix_tokens: int = int(
            getattr(model_runner.server_args, "cascade_min_prefix_tokens", 128)
        )
        self.cascade_min_batch_size: int = int(
            getattr(model_runner.server_args, "cascade_min_batch_size", 4)
        )
        if self.cascade_min_prefix_tokens < 1:
            self.cascade_min_prefix_tokens = 1
        if self.cascade_min_batch_size < 2:
            self.cascade_min_batch_size = 2

        # Tracks whether the current step is inside CG capture/replay (set by
        # the CG hooks below; reset by eager init).
        self._in_cuda_graph: bool = False

        # Parent's KV pool: SGLang treats each slot as a 1-page row in the
        # flashinfer wrapper (page_size=1). We mirror that here for the
        # cascade wrapper so kv_indices are slot ids the same way.
        self.cascade_page_size: int = 1
        self.num_qo_heads: int = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads_local: int = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim_local: int = model_runner.model_config.head_dim
        self.q_dtype = model_runner.dtype
        self.kv_dtype = model_runner.kv_cache_dtype

        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.req_to_token_stride = self.req_to_token.shape[1]
        self._device = model_runner.device

        # Level-0 common-prefix detection intentionally has no fixed token cap.
        # It scans up to the shortest active sequence so a full-batch shared
        # prefix can be represented entirely as level 0.

        # Eager-mode cascade wrapper. Use_cuda_graph=False; plan allocates
        # scheduler state per call (acceptable for eager since plan runs
        # per step anyway).
        self._cascade_decode_wrapper: Optional[MultiLevelCascadeAttentionWrapper] = None
        if is_flashinfer_available():
            self._cascade_decode_wrapper = MultiLevelCascadeAttentionWrapper(
                num_levels=3,
                float_workspace_buffer=self.workspace_buffer,
                kv_layout="NHD",
                use_cuda_graph=False,
            )

        # Eager plan stash. Set by init_forward_metadata; read by
        # forward_decode. ``None`` means eager cascade did not arm.
        self._cascade_plan: Optional[_CascadePlanState] = None

        # ---- CG-mode state ----
        # Per-bs wrappers + pre-allocated buffers. Filled lazily during
        # init_forward_metadata_out_graph(in_capture=True) (called once per
        # cuda_graph_bs by the runner). Each entry holds a wrapper sized
        # to one specific captured batch size; FlashInfer's CG-mode
        # contract enforces fixed batch_size across plan() calls.
        self._cg_cascade_wrappers: dict = {}  # bs -> MultiLevelCascadeAttentionWrapper
        self._cg_cascade_buffers: dict = {}  # bs -> dict of pre-alloc tensors
        # Per-replay plan stash for CG path (separate from eager path so
        # the two arms cannot interfere). Set by replay hook; read by
        # forward_decode when ``_in_cuda_graph`` is true.
        self._cg_cascade_plan: Optional[_CascadePlanState] = None
        # CG buffer sizing (set in init_cuda_graph_state).
        self._cg_max_bs: int = 0
        # Worst-case shared prefix length: cap by (max_context_len, scan_cap).
        # Using max_context_len ensures level-0 indices buffer can hold any
        # plausible prefix within the model's context window.
        self._cg_max_shared_pages: int = 0
        # Worst-case unique slots per request: max_context_len.
        self._cg_max_pages_per_req: int = 0
        self._last_cg_cascade_plan_failure: str = ""

        # Toggle: SGLANG_CASCADE_DISABLE_CUDA_GRAPH=1 forces eager-only cascade
        # even when CG is enabled (useful for debugging and bisection).
        self._cg_disabled: bool = (
            os.environ.get("SGLANG_CASCADE_DISABLE_CUDA_GRAPH", "0") == "1"
        )
        self._force_no_prefix_cascade: bool = (
            os.environ.get("SGLANG_CASCADE_FORCE_NO_PREFIX", "0") == "1"
        )

        # Debug counters. Set ``SGLANG_CASCADE_DEBUG=1`` to log per-step
        # decisions. Used by tests to confirm fire/skip behavior.
        self._dbg_enabled: bool = os.environ.get("SGLANG_CASCADE_DEBUG", "0") == "1"
        self._dbg_total_decode_steps: int = 0
        self._dbg_cascade_fired: int = 0  # eager fires (>= threshold)
        self._dbg_cascade_fired_cg: int = 0  # CG-mode fires (>= threshold)
        self._dbg_cascade_run_cg: int = (
            0  # CG-mode actually ran (incl. below-threshold)
        )
        self._dbg_skip_below_bs: int = 0
        self._dbg_skip_below_prefix: int = 0
        self._dbg_skip_below_prefix_cg: int = 0  # CG ran but threshold not met
        self._dbg_skip_in_cg: int = 0
        self._dbg_skip_not_decode: int = 0
        self._dbg_cg_graph_key_log_count: int = 0
        self._dbg_cg_graph_key_log_limit: int = int(
            os.environ.get("SGLANG_CASCADE_DEBUG_CG_GRAPH_KEY_LIMIT", "64")
        )
        self._last_cg_cascade_plan_debug: dict = {}
        # Set SGLANG_CASCADE_DEBUG_KERNEL_INPUTS=1 to inspect the eager
        # FlashInfer cascade plan/run inputs. This intentionally does not log
        # under CUDA graph capture/replay because those code paths require
        # stable buffer addresses and logs would add noisy synchronizations.
        self._dbg_kernel_inputs_enabled: bool = (
            os.environ.get("SGLANG_CASCADE_DEBUG_KERNEL_INPUTS", "0") == "1"
        )
        self._dbg_kernel_inputs_limit: int = int(
            os.environ.get("SGLANG_CASCADE_DEBUG_KERNEL_INPUTS_LIMIT", "16")
        )
        self._dbg_kernel_inputs_count: int = 0
        self._dbg_kernel_inputs_sample: int = int(
            os.environ.get("SGLANG_CASCADE_DEBUG_KERNEL_INPUTS_SAMPLE", "16")
        )
        self._dbg_kernel_inputs_plan_only: bool = (
            os.environ.get("SGLANG_CASCADE_DEBUG_KERNEL_INPUTS_PLAN_ONLY", "0") == "1"
        )
        self._auto_detect_level1_enabled: bool = (
            os.environ.get("SGLANG_CASCADE_AUTO_DETECT_LEVEL1", "0") == "1"
        )
        self._auto_detect_scan_cap: int = int(
            os.environ.get("SGLANG_CASCADE_AUTO_DETECT_SCAN_CAP", "32768")
        )
        if self._auto_detect_scan_cap <= 0:
            self._auto_detect_scan_cap = int(self.max_context_len)

        logger.info(
            "FlashInferCascadeAttnBackend initialized "
            "(min_prefix_tokens=%d, min_batch_size=%d, num_qo_heads=%d, "
            "num_kv_heads=%d, head_dim=%d, cg_disabled=%s, "
            "debug_kernel_inputs=%s, debug_kernel_inputs_limit=%d, "
            "auto_detect_level1=%s, auto_detect_scan_cap=%d, "
            "force_no_prefix_cascade=%s)",
            self.cascade_min_prefix_tokens,
            self.cascade_min_batch_size,
            self.num_qo_heads,
            self.num_kv_heads_local,
            self.head_dim_local,
            self._cg_disabled,
            self._dbg_kernel_inputs_enabled,
            self._dbg_kernel_inputs_limit,
            self._auto_detect_level1_enabled,
            self._auto_detect_scan_cap,
            self._force_no_prefix_cascade,
        )

    def cascade_debug_counters(self) -> dict:
        """Snapshot of debug counters; used by tests to assert fire/skip
        behavior. Always available regardless of ``SGLANG_CASCADE_DEBUG``.
        """
        return {
            "total_decode_steps": self._dbg_total_decode_steps,
            "cascade_fired": self._dbg_cascade_fired,
            "cascade_fired_cg": self._dbg_cascade_fired_cg,
            "cascade_run_cg": self._dbg_cascade_run_cg,
            "skip_below_bs": self._dbg_skip_below_bs,
            "skip_below_prefix": self._dbg_skip_below_prefix,
            "skip_below_prefix_cg": self._dbg_skip_below_prefix_cg,
            "skip_in_cg": self._dbg_skip_in_cg,
            "skip_not_decode": self._dbg_skip_not_decode,
            "last_cg_cascade_plan_failure": self._last_cg_cascade_plan_failure,
        }

    def get_cuda_graph_seq_len_fill_value(self):
        # Cascade metadata treats seq_len<=1 as a non-cascade request in
        # non-force mode. CUDA graph padding slots are synthetic, so keep them
        # at length 2 to avoid letting padding alone disable replay planning.
        return 2

    @staticmethod
    def _cascade_graph_variant_label(
        layout_kind: str, group_bucket: Optional[int] = None
    ) -> str:
        if layout_kind not in ("level1_only", "level0_level1"):
            return f"cascade:{layout_kind}"
        return f"cascade:{layout_kind}:g{int(group_bucket)}"

    @staticmethod
    def _parse_cascade_graph_variant_label(
        label: Optional[str],
    ) -> tuple[str, Optional[int]]:
        if not label:
            return "no_prefix", None
        for part in str(label).split("|"):
            if not part.startswith("cascade:"):
                continue
            pieces = part.split(":")
            if len(pieces) == 2:
                return pieces[1], None
            if len(pieces) != 3 or not pieces[2].startswith("g"):
                continue
            try:
                return pieces[1], int(pieces[2][1:])
            except ValueError:
                continue
        return "no_prefix", None

    def _should_cg_no_prefix_fallback(self, variant_label: Optional[str]) -> bool:
        layout_kind, _ = self._parse_cascade_graph_variant_label(variant_label)
        return layout_kind == "no_prefix" and not self._force_no_prefix_cascade

    @staticmethod
    def _cascade_group_count_buckets(bs: int) -> list[int]:
        half = max(1, (bs + 1) // 2)
        three_quarter = max(1, (3 * bs + 3) // 4)
        return list(dict.fromkeys([half, three_quarter, bs]))

    def _bucketize_cascade_group_count(self, num_groups: int, bs: int) -> int:
        for bucket in self._cascade_group_count_buckets(bs):
            if num_groups <= bucket:
                return bucket
        return bs

    def _classify_cascade_graph_variant_from_meta(self, meta: dict, bs: int) -> str:
        system_prefix = int(meta["system_prefix"])
        groups = meta["groups"]
        has_level0 = system_prefix > 0
        has_effective_level1 = any(
            len(group["members"]) > 1
            and int(group["shared_prefix"]) > system_prefix
            for group in groups
        )

        if not has_level0 and not has_effective_level1:
            layout_kind = "no_prefix"
        elif has_level0 and not has_effective_level1:
            layout_kind = "level0_only"
        elif not has_level0 and has_effective_level1:
            layout_kind = "level1_only"
        else:
            layout_kind = "level0_level1"

        group_bucket = None
        if layout_kind in ("level1_only", "level0_level1"):
            group_bucket = self._bucketize_cascade_group_count(len(groups), bs)
        return self._cascade_graph_variant_label(layout_kind, group_bucket)

    def get_cuda_graph_capture_variant_labels(self, bs: int) -> list[str]:
        """Cascade-specific graph variants to capture for one padded bs.

        This intentionally keeps the variant set small. Online shared-prefix
        case2/case3 use pair groups, so g=bs/2 covers the primary experiment;
        no-prefix uses singleton groups.
        """
        labels = [self._cascade_graph_variant_label("no_prefix")]
        labels.append(self._cascade_graph_variant_label("level0_only"))
        if bs >= 2:
            for num_groups in self._cascade_group_count_buckets(bs):
                labels.append(
                    self._cascade_graph_variant_label("level1_only", num_groups)
                )
                labels.append(
                    self._cascade_graph_variant_label(
                        "level0_level1", num_groups
                    )
                )
        # Preserve order while removing duplicates for bs=1/2 corner cases.
        return list(dict.fromkeys(labels))

    def get_cuda_graph_variant_label(
        self,
        forward_batch: ForwardBatch,
        cuda_graph_bs: Optional[int] = None,
    ) -> Optional[str]:
        bs = int(getattr(forward_batch, "batch_size", 0) or 0)
        if bs <= 0:
            return None
        variant_bs = int(cuda_graph_bs) if cuda_graph_bs is not None else bs
        common = 0
        try:
            common = self._detect_common_prefix_from_rpi(
                bs, forward_batch.req_pool_indices, forward_batch.seq_lens_cpu
            )
        except Exception:
            common = 0
        meta = self._build_three_level_metadata(
            bs=bs,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens_cpu=getattr(forward_batch, "seq_lens_cpu", None),
            rids=getattr(forward_batch, "rids", None),
            prefix_ref_rids=getattr(forward_batch, "cascade_prefix_ref_rids", None),
            shared_prefix_lens=getattr(
                forward_batch, "cascade_shared_prefix_lens_cpu", None
            ),
            system_prefix_lens=getattr(
                forward_batch, "cascade_system_prefix_lens_cpu", None
            ),
            fallback_common_prefix=common,
            fallback_seq_lens=getattr(forward_batch, "seq_lens", None),
        )
        if meta is None:
            return self._cascade_graph_variant_label("no_prefix")
        return self._classify_cascade_graph_variant_from_meta(meta, variant_bs)

    def prepare_cuda_graph_capture_forward_batch(
        self, forward_batch: ForwardBatch, variant_label: Optional[str]
    ) -> None:
        """Populate cascade metadata on capture dummy batches.

        The decode runner creates generic dummy batches; this hook makes their
        metadata match the graph variant being captured.
        """
        layout_kind, group_bucket = self._parse_cascade_graph_variant_label(
            variant_label
        )
        bs = int(forward_batch.batch_size)
        if group_bucket is None or group_bucket <= 0:
            group_bucket = bs
        group_bucket = max(1, min(group_bucket, bs))

        if layout_kind == "level0_level1":
            seq_len = 3
            system_prefix = 1
            shared_prefix = 2
        elif layout_kind == "level0_only":
            seq_len = 2
            system_prefix = 1
            shared_prefix = 1
        elif layout_kind == "level1_only":
            seq_len = 2
            system_prefix = 0
            shared_prefix = 1
        else:
            seq_len = 2
            system_prefix = 0
            shared_prefix = 0

        forward_batch.seq_lens.fill_(seq_len)
        if forward_batch.seq_lens_cpu is not None:
            forward_batch.seq_lens_cpu.fill_(seq_len)
        forward_batch.seq_lens_sum = seq_len * bs

        rids = [f"cg_req{i}" for i in range(bs)]
        refs: list[Optional[str]] = [None] * bs
        shared_lens: list[Optional[int]] = [shared_prefix] * bs
        system_lens: list[Optional[int]] = [system_prefix] * bs

        if layout_kind in ("level1_only", "level0_level1"):
            groups = [[] for _ in range(group_bucket)]
            for i in range(bs):
                groups[i % group_bucket].append(i)
            for members in groups:
                if not members:
                    continue
                anchor = members[0]
                for member in members[1:]:
                    refs[member] = rids[anchor]
        elif layout_kind == "no_prefix":
            shared_lens = [0] * bs
            system_lens = [0] * bs

        forward_batch.rids = rids
        forward_batch.cascade_prefix_ref_rids = refs
        forward_batch.cascade_shared_prefix_lens_cpu = shared_lens
        forward_batch.cascade_system_prefix_lens_cpu = system_lens

    def _set_cg_cascade_plan_failure(self, reason: str) -> bool:
        self._last_cg_cascade_plan_failure = reason
        if self._dbg_enabled:
            logger.warning("CG cascade plan failure reason: %s", reason)
        return False

    def _should_log_eager_kernel_inputs(self) -> bool:
        if not self._dbg_kernel_inputs_enabled or self._in_cuda_graph:
            return False
        return (
            self._dbg_kernel_inputs_limit < 0
            or self._dbg_kernel_inputs_count < self._dbg_kernel_inputs_limit
        )

    def _tensor_debug_summary(
        self,
        tensor: Optional[torch.Tensor],
        active_numel: Optional[int] = None,
    ) -> dict:
        if tensor is None:
            return {"is_none": True}

        numel = int(tensor.numel())
        if active_numel is None:
            active_numel = numel
        active_numel = max(0, min(int(active_numel), numel))
        sample_n = max(0, min(self._dbg_kernel_inputs_sample, active_numel))

        flat = tensor.detach().reshape(-1)
        head = flat[:sample_n].cpu().tolist() if sample_n > 0 else []
        tail = []
        if active_numel > sample_n:
            tail = flat[active_numel - sample_n : active_numel].cpu().tolist()

        return {
            "shape": tuple(tensor.shape),
            "stride": tuple(tensor.stride()),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "numel": numel,
            "active_numel": active_numel,
            "head": head,
            "tail": tail,
        }

    def _log_eager_kernel_inputs(self, message: str, payload: dict) -> None:
        if not self._should_log_eager_kernel_inputs():
            return
        self._dbg_kernel_inputs_count += 1
        logger.info("%s #%d: %s", message, self._dbg_kernel_inputs_count, payload)

    def _log_cg_graph_key_decision(
        self,
        phase: str,
        bs: int,
        graph_key: Any,
        variant_label: Optional[str],
        common_prefix_tokens: int,
    ) -> None:
        if not self._dbg_enabled:
            return
        if (
            self._dbg_cg_graph_key_log_limit >= 0
            and self._dbg_cg_graph_key_log_count
            >= self._dbg_cg_graph_key_log_limit
        ):
            return
        self._dbg_cg_graph_key_log_count += 1
        payload = {
            "phase": phase,
            "bs": int(bs),
            "graph_key": graph_key,
            "variant_label": variant_label,
            "common_prefix_tokens": int(common_prefix_tokens),
        }
        payload.update(self._last_cg_cascade_plan_debug)
        logger.info(
            "CG cascade graph key decision #%d: %s",
            self._dbg_cg_graph_key_log_count,
            payload,
        )

    def _log_cg_no_prefix_fallback(
        self,
        phase: str,
        bs: int,
        graph_key: Any,
        variant_label: Optional[str],
    ) -> None:
        if not self._dbg_enabled:
            return
        if (
            self._dbg_cg_graph_key_log_limit >= 0
            and self._dbg_cg_graph_key_log_count
            >= self._dbg_cg_graph_key_log_limit
        ):
            return
        self._dbg_cg_graph_key_log_count += 1
        logger.info(
            "CG cascade graph key decision #%d: %s",
            self._dbg_cg_graph_key_log_count,
            {
                "phase": phase,
                "bs": int(bs),
                "graph_key": graph_key,
                "variant_label": variant_label,
                "path": "flashinfer_parent",
                "reason": "no_prefix_fallback",
            },
        )

    # ------------------------------------------------------------------
    # Detection helpers (host-side; used in both eager and CG paths)
    # ------------------------------------------------------------------

    def _detect_common_prefix_from_rpi(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
    ) -> int:
        """CG-friendly detection: walks the leading slot indices using only
        ``req_pool_indices`` and ``seq_lens_cpu`` (no forward_batch).

        Returns the count of leading positions where all requests share the
        same slot index, capped at ``min_seq - 1`` so each request keeps at
        least 1 unique slot for its current decode token.
        """
        if bs < 2:
            return 0
        if seq_lens_cpu is None:
            seq_lens_cpu = req_pool_indices.new_empty(0)  # placeholder
        if seq_lens_cpu.numel() == 0:
            return 0
        min_seq = int(seq_lens_cpu[:bs].min().item())
        if min_seq <= 1:
            return 0
        scan_n = min_seq
        # Compare/reduce entirely on-device and sync only a single scalar
        # back to host (instead of copying the whole [bs, scan_n] slice).
        leading = self.req_to_token[req_pool_indices[:bs].long(), :scan_n]
        # mismatch[j] is True if any request's slot j differs from request 0's.
        mismatch = (leading != leading[0:1]).any(dim=0)
        # Append a True sentinel at index scan_n so argmax always resolves to
        # the first divergence -- or to scan_n itself when fully shared. This
        # keeps the whole reduction on-device with one scalar sync.
        sentinel = torch.ones(1, dtype=torch.bool, device=mismatch.device)
        common = int(torch.cat([mismatch, sentinel]).to(torch.uint8).argmax().item())
        common = min(common, min_seq - 1)
        return max(0, common)

    def _detect_common_prefix_tokens(self, forward_batch: ForwardBatch, bs: int) -> int:
        """Eager-mode detection wrapper -- forwards to the CG-friendly impl
        using the forward_batch's seq_lens_cpu / req_pool_indices.
        """
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = forward_batch.seq_lens.cpu()
        return self._detect_common_prefix_from_rpi(
            bs, forward_batch.req_pool_indices, seq_lens_cpu
        )

    def _get_seq_lens_list(
        self,
        bs: int,
        seq_lens_cpu: Optional[torch.Tensor],
        fallback_device_tensor: Optional[torch.Tensor] = None,
    ) -> list[int]:
        if seq_lens_cpu is None:
            if fallback_device_tensor is None:
                return [1] * bs
            seq_lens_cpu = fallback_device_tensor.cpu()
        return [int(x) for x in seq_lens_cpu[:bs].tolist()]

    def _longest_shared_prefix_for_members(
        self,
        member_indices: list[int],
        rpi_list: list[int],
        seq_lens: list[int],
        scan_cap: int,
    ) -> tuple[int, bool]:
        if not member_indices:
            return 0, False
        min_seq = min(seq_lens[i] for i in member_indices)
        if min_seq <= 1:
            return 0, False
        scan_n = min(min_seq, scan_cap)
        if scan_n <= 0:
            return 0, False

        member_rpis = torch.tensor(
            [int(rpi_list[i]) for i in member_indices],
            dtype=torch.long,
            device=self.req_to_token.device,
        )
        leading = self.req_to_token[member_rpis, :scan_n]
        mismatch = (leading != leading[0:1]).any(dim=0)
        sentinel = torch.ones(1, dtype=torch.bool, device=mismatch.device)
        common = int(torch.cat([mismatch, sentinel]).to(torch.uint8).argmax().item())
        common = min(common, min_seq - 1)
        cap_limited = common == scan_n and scan_n < min_seq - 1
        return max(0, common), cap_limited

    def _build_auto_detected_three_level_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: list[int],
        system_prefix: int,
    ) -> Optional[dict]:
        """Infer Level-1 groups from req_to_token slot equality.

        This path is used only when every request is missing cascade metadata.
        It keeps Level 0 as the already-detected whole-batch common prefix and
        groups requests whose next slot after Level 0 is shared.
        """
        if bs <= 0:
            return None

        rpi_list = req_pool_indices[:bs].cpu().tolist()
        system_prefix = max(0, min(int(system_prefix), min(seq_lens) - 1))

        candidate_groups: dict[int, list[int]] = {}
        singleton_indices: list[int] = []
        for i in range(bs):
            if seq_lens[i] <= system_prefix + 1:
                singleton_indices.append(i)
                continue
            key = int(
                self.req_to_token[int(rpi_list[i]), system_prefix]
                .detach()
                .cpu()
                .item()
            )
            candidate_groups.setdefault(key, []).append(i)

        group_specs: list[tuple[int, list[int], int]] = []
        per_req_shared = [system_prefix] * bs
        cap_limited = False

        for members in candidate_groups.values():
            members = sorted(members)
            if len(members) < 2:
                singleton_indices.extend(members)
                continue
            shared_len, limited = self._longest_shared_prefix_for_members(
                members,
                rpi_list,
                seq_lens,
                self._auto_detect_scan_cap,
            )
            cap_limited = cap_limited or limited
            shared_len = max(system_prefix, shared_len)
            if shared_len <= system_prefix:
                singleton_indices.extend(members)
                continue
            anchor_idx = min(members)
            for idx in members:
                per_req_shared[idx] = min(shared_len, seq_lens[idx] - 1)
            group_specs.append((anchor_idx, members, shared_len))

        for idx in singleton_indices:
            group_specs.append((idx, [idx], system_prefix))

        group_specs.sort(key=lambda item: min(item[1]))
        groups = []
        perm_list: list[int] = []
        q_indptr_l1_cpu = [0]
        for anchor_idx, members, shared_len in group_specs:
            q_start = len(perm_list)
            perm_list.extend(members)
            q_indptr_l1_cpu.append(len(perm_list))
            groups.append(
                {
                    "anchor_idx": int(anchor_idx),
                    "members": members,
                    "shared_prefix": int(shared_len),
                    "q_start": q_start,
                    "q_end": len(perm_list),
                }
            )

        inv_perm_list = [0] * bs
        for new_idx, old_idx in enumerate(perm_list):
            inv_perm_list[old_idx] = new_idx

        return {
            "seq_lens": seq_lens,
            "system_prefix": system_prefix,
            "groups": groups,
            "per_req_shared": per_req_shared,
            "perm": perm_list,
            "inv_perm": inv_perm_list,
            "q_indptr_l1_cpu": q_indptr_l1_cpu,
            "metadata_enabled": False,
            "layout_source": "auto_detect",
            "auto_detect_cap_limited": cap_limited,
            "auto_detect_scan_cap": self._auto_detect_scan_cap,
        }

    def _build_three_level_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        rids: Optional[list[str]] = None,
        prefix_ref_rids: Optional[list[Optional[str]]] = None,
        shared_prefix_lens: Optional[list[Optional[int]]] = None,
        system_prefix_lens: Optional[list[Optional[int]]] = None,
        fallback_common_prefix: int = 0,
        fallback_seq_lens: Optional[torch.Tensor] = None,
    ) -> Optional[dict]:
        """Normalize request metadata into a fixed three-level cascade layout.

        The returned layout is group-major: all members of a Level-1 group are
        contiguous after applying ``perm``. Requests without metadata become
        singleton groups with a zero-length Level-1 segment.
        """
        if bs <= 0:
            return None

        seq_lens = self._get_seq_lens_list(bs, seq_lens_cpu, fallback_seq_lens)
        min_decode_len = min(seq_lens) if seq_lens else 0
        if min_decode_len <= 0:
            return None
        if min_decode_len <= 1 and not self._force_no_prefix_cascade:
            return None

        def _pad(values, default):
            out = list(values[:bs]) if values is not None else []
            if len(out) < bs:
                out.extend([default] * (bs - len(out)))
            return out

        def _to_int_or_none(value) -> Optional[int]:
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        rids = _pad(rids, None)
        rids = [str(rid) if rid is not None else str(i) for i, rid in enumerate(rids)]
        prefix_ref_rids = _pad(prefix_ref_rids, None)
        shared_prefix_lens = [_to_int_or_none(x) for x in _pad(shared_prefix_lens, None)]
        system_prefix_lens = [_to_int_or_none(x) for x in _pad(system_prefix_lens, None)]
        has_metadata = (
            any(x is not None for x in shared_prefix_lens)
            or any(x is not None for x in prefix_ref_rids)
            or any(x is not None for x in system_prefix_lens)
        )

        specified_system_lens = [x for x in system_prefix_lens if x is not None and x >= 0]
        system_prefix = (
            min(specified_system_lens)
            if specified_system_lens
            else int(fallback_common_prefix)
        )
        system_prefix = max(0, min(system_prefix, min_decode_len - 1))

        if self._auto_detect_level1_enabled and not has_metadata:
            return self._build_auto_detected_three_level_metadata(
                bs=bs,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                system_prefix=system_prefix,
            )

        rid_to_idx = {rid: i for i, rid in enumerate(rids)}
        groups_by_key: dict[tuple[int, int], list[int]] = {}
        per_req_shared: list[int] = []
        for i in range(bs):
            max_shared_i = max(0, seq_lens[i] - 1)
            declared_shared = shared_prefix_lens[i]
            if declared_shared is None:
                shared_i = system_prefix
            else:
                shared_i = int(declared_shared)
            shared_i = max(system_prefix, min(shared_i, max_shared_i))

            ref_rid = prefix_ref_rids[i]
            anchor_idx = rid_to_idx.get(ref_rid, i) if ref_rid is not None else i
            if shared_i == system_prefix:
                # No Level-1 middle segment; keep it independent to avoid
                # incorrectly grouping unrelated unique tails.
                anchor_idx = i
            anchor_max_shared = max(0, seq_lens[anchor_idx] - 1)
            shared_i = max(system_prefix, min(shared_i, anchor_max_shared))
            per_req_shared.append(shared_i)
            groups_by_key.setdefault((anchor_idx, shared_i), []).append(i)

        group_items = sorted(groups_by_key.items(), key=lambda item: min(item[1]))
        groups = []
        perm_list: list[int] = []
        q_indptr_l1_cpu = [0]
        for (anchor_idx, shared_i), members in group_items:
            members = sorted(members)
            q_start = len(perm_list)
            perm_list.extend(members)
            q_indptr_l1_cpu.append(len(perm_list))
            groups.append(
                {
                    "anchor_idx": int(anchor_idx),
                    "members": members,
                    "shared_prefix": int(shared_i),
                    "q_start": q_start,
                    "q_end": len(perm_list),
                }
            )

        inv_perm_list = [0] * bs
        for new_idx, old_idx in enumerate(perm_list):
            inv_perm_list[old_idx] = new_idx

        return {
            "seq_lens": seq_lens,
            "system_prefix": system_prefix,
            "groups": groups,
            "per_req_shared": per_req_shared,
            "perm": perm_list,
            "inv_perm": inv_perm_list,
            "q_indptr_l1_cpu": q_indptr_l1_cpu,
            "metadata_enabled": has_metadata,
            "layout_source": "metadata" if has_metadata else "fallback_singleton",
            "auto_detect_cap_limited": False,
            "auto_detect_scan_cap": self._auto_detect_scan_cap,
        }

    # ------------------------------------------------------------------
    # Eager-mode cascade plan
    # ------------------------------------------------------------------

    def _build_cascade_plan_args(
        self,
        forward_batch: ForwardBatch,
        bs: int,
        common_prefix_tokens: int,
    ):
        """Eager-mode three-level plan builder."""
        device = forward_batch.input_ids.device
        meta = self._build_three_level_metadata(
            bs=bs,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            rids=forward_batch.rids,
            prefix_ref_rids=getattr(forward_batch, "cascade_prefix_ref_rids", None),
            shared_prefix_lens=getattr(
                forward_batch, "cascade_shared_prefix_lens_cpu", None
            ),
            system_prefix_lens=getattr(
                forward_batch, "cascade_system_prefix_lens_cpu", None
            ),
            fallback_common_prefix=common_prefix_tokens,
            fallback_seq_lens=forward_batch.seq_lens,
        )
        if meta is None:
            return None

        seq_lens_list = meta["seq_lens"]
        system_prefix = meta["system_prefix"]
        groups = meta["groups"]
        per_req_shared = meta["per_req_shared"]
        perm_list = meta["perm"]
        inv_perm_list = meta["inv_perm"]
        rpi_list = forward_batch.req_pool_indices[:bs].cpu().tolist()

        qo_indptr_l0 = torch.tensor([0, bs], dtype=torch.int32, device=device)
        kv_indptr_l0 = torch.tensor([0, system_prefix], dtype=torch.int32, device=device)
        kv_indices_l0 = torch.zeros(
            max(1, system_prefix), dtype=torch.int32, device=device
        )
        if system_prefix > 0:
            kv_indices_l0[:system_prefix].copy_(
                self.req_to_token[int(rpi_list[0]), :system_prefix].to(torch.int32),
                non_blocking=True,
            )
        last_page_len_l0 = torch.tensor([1], dtype=torch.int32, device=device)
        if system_prefix == 0:
            last_page_len_l0.fill_(0)

        qo_indptr_l1 = torch.tensor(
            meta["q_indptr_l1_cpu"], dtype=torch.int32, device=device
        )
        kv_indptr_l1_cpu = [0]
        kv_indices_l1_parts = []
        for group in groups:
            middle_len = max(0, group["shared_prefix"] - system_prefix)
            if middle_len > 0:
                anchor_rpi = int(rpi_list[group["anchor_idx"]])
                kv_indices_l1_parts.append(
                    self.req_to_token[
                        anchor_rpi, system_prefix : group["shared_prefix"]
                    ].to(torch.int32)
                )
            kv_indptr_l1_cpu.append(kv_indptr_l1_cpu[-1] + middle_len)
        total_middle = kv_indptr_l1_cpu[-1]
        kv_indptr_l1 = torch.tensor(kv_indptr_l1_cpu, dtype=torch.int32, device=device)
        if kv_indices_l1_parts:
            kv_indices_l1 = torch.cat(kv_indices_l1_parts)
        else:
            kv_indices_l1 = torch.zeros(1, dtype=torch.int32, device=device)
        last_page_len_l1 = torch.ones(len(groups), dtype=torch.int32, device=device)
        for i, group in enumerate(groups):
            if group["shared_prefix"] == system_prefix:
                last_page_len_l1[i] = 0
        if (
            meta.get("layout_source") == "auto_detect"
            and system_prefix < self.cascade_min_prefix_tokens
            and total_middle <= 0
            and not self._force_no_prefix_cascade
        ):
            return None

        qo_indptr_l2 = torch.arange(bs + 1, dtype=torch.int32, device=device)
        kv_indptr_l2_cpu = [0]
        kv_indices_l2_parts = []
        for old_idx in perm_list:
            shared_i = per_req_shared[old_idx]
            tail_len = max(0, int(seq_lens_list[old_idx]) - shared_i)
            if tail_len > 0:
                rpi_i = int(rpi_list[old_idx])
                kv_indices_l2_parts.append(
                    self.req_to_token[
                        rpi_i, shared_i : shared_i + tail_len
                    ].to(torch.int32)
                )
            kv_indptr_l2_cpu.append(kv_indptr_l2_cpu[-1] + tail_len)
        total_tail = kv_indptr_l2_cpu[-1]
        if total_tail <= 0:
            return None
        kv_indptr_l2 = torch.tensor(kv_indptr_l2_cpu, dtype=torch.int32, device=device)
        kv_indices_l2 = (
            torch.cat(kv_indices_l2_parts)
            if kv_indices_l2_parts
            else torch.zeros(1, dtype=torch.int32, device=device)
        )
        last_page_len_l2 = torch.ones(bs, dtype=torch.int32, device=device)
        perm = torch.tensor(perm_list, dtype=torch.long, device=device)
        inv_perm = torch.tensor(inv_perm_list, dtype=torch.long, device=device)

        self._log_eager_kernel_inputs(
            "Cascade eager plan kernel inputs",
            {
                "bs": bs,
                "fallback_common_prefix_tokens": common_prefix_tokens,
                "system_prefix_tokens": system_prefix,
                "seq_lens": seq_lens_list,
                "req_pool_indices": rpi_list,
                "per_req_shared": per_req_shared,
                "groups": groups,
                "layout_source": meta.get("layout_source", "unknown"),
                "auto_detect_scan_cap": meta.get("auto_detect_scan_cap"),
                "auto_detect_cap_limited": meta.get("auto_detect_cap_limited"),
                "total_middle": total_middle,
                "total_tail": total_tail,
                "perm": perm_list,
                "inv_perm": inv_perm_list,
                "qo_indptr_l0": self._tensor_debug_summary(qo_indptr_l0),
                "qo_indptr_l1": self._tensor_debug_summary(qo_indptr_l1),
                "qo_indptr_l2": self._tensor_debug_summary(qo_indptr_l2),
                "kv_indptr_l0": self._tensor_debug_summary(kv_indptr_l0),
                "kv_indptr_l1": self._tensor_debug_summary(kv_indptr_l1),
                "kv_indptr_l2": self._tensor_debug_summary(kv_indptr_l2),
                "kv_indices_l0": self._tensor_debug_summary(
                    kv_indices_l0, max(1, system_prefix)
                ),
                "kv_indices_l1": self._tensor_debug_summary(
                    kv_indices_l1, max(1, total_middle)
                ),
                "kv_indices_l2": self._tensor_debug_summary(
                    kv_indices_l2, max(1, total_tail)
                ),
                "last_page_len_l0": self._tensor_debug_summary(last_page_len_l0),
                "last_page_len_l1": self._tensor_debug_summary(last_page_len_l1),
                "last_page_len_l2": self._tensor_debug_summary(last_page_len_l2),
                "num_qo_heads": self.num_qo_heads,
                "num_kv_heads": self.num_kv_heads_local,
                "head_dim": self.head_dim_local,
                "page_size": self.cascade_page_size,
                "causal": False,
                "pos_encoding_mode": "NONE",
                "q_data_type": str(self.q_dtype),
                "kv_data_type": str(self.kv_dtype),
            },
        )

        try:
            self._cascade_decode_wrapper.plan(
                qo_indptr_arr=[qo_indptr_l0, qo_indptr_l1, qo_indptr_l2],
                paged_kv_indptr_arr=[kv_indptr_l0, kv_indptr_l1, kv_indptr_l2],
                paged_kv_indices_arr=[kv_indices_l0, kv_indices_l1, kv_indices_l2],
                paged_kv_last_page_len=[
                    last_page_len_l0,
                    last_page_len_l1,
                    last_page_len_l2,
                ],
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads_local,
                head_dim=self.head_dim_local,
                page_size=self.cascade_page_size,
                causal=False,
                pos_encoding_mode="NONE",
                q_data_type=self.q_dtype,
                kv_data_type=self.kv_dtype,
            )
        except Exception as e:
            if self._dbg_enabled:
                logger.warning("Cascade plan failed, falling through: %s", e)
            return None

        return _CascadePlanState(
            common_prefix_tokens=system_prefix, bs=bs, perm=perm, inv_perm=inv_perm
        )

    # ------------------------------------------------------------------
    # CG-mode cascade plan
    # ------------------------------------------------------------------

    def _allocate_cg_cascade_for_key(self, graph_key: Any, bs: int) -> None:
        """Allocate per-bs cascade wrapper + indptr/indices buffers. Called
        lazily on first capture for each cuda_graph_bs entry.

        Buffer sizing per level (FlashInfer 0.6.8.post1 contract):
            Level 0 (shared, 1 merged group):
                qo_indptr [2], paged_kv_indptr [2],
                paged_kv_indices [_cg_max_shared_pages],
                paged_kv_last_page_len [1]
            Level 1 (bs unique tails):
                qo_indptr [bs + 1], paged_kv_indptr [bs + 1],
                paged_kv_indices [bs * _cg_max_pages_per_req],
                paged_kv_last_page_len [bs]
        """
        if graph_key in self._cg_cascade_wrappers:
            return
        d = self._device

        # Level 1 group layout can change per replay, so qo_indptr_l1 is
        # mutable. Level 2 is always one query per reordered request.
        qo_indptr_l0 = torch.tensor([0, bs], dtype=torch.int32, device=d)
        qo_indptr_l1 = torch.zeros(bs + 1, dtype=torch.int32, device=d)
        qo_indptr_l2 = torch.arange(bs + 1, dtype=torch.int32, device=d)
        kv_indptr_l0 = torch.zeros(2, dtype=torch.int32, device=d)
        kv_indptr_l1 = torch.zeros(bs + 1, dtype=torch.int32, device=d)
        kv_indptr_l2 = torch.zeros(bs + 1, dtype=torch.int32, device=d)
        kv_indices_l0 = torch.zeros(
            self._cg_max_shared_pages, dtype=torch.int32, device=d
        )
        # Level-1 has at most bs groups, each up to max_context_len slots.
        kv_indices_l1 = torch.zeros(
            bs * self._cg_max_pages_per_req, dtype=torch.int32, device=d
        )
        # Level-2 indices: bs requests, each up to max_context_len slots.
        kv_indices_l2 = torch.zeros(
            bs * self._cg_max_pages_per_req, dtype=torch.int32, device=d
        )
        last_page_l0 = torch.ones(1, dtype=torch.int32, device=d)
        last_page_l1 = torch.ones(bs, dtype=torch.int32, device=d)
        last_page_l2 = torch.ones(bs, dtype=torch.int32, device=d)
        perm = torch.arange(bs, dtype=torch.long, device=d)
        inv_perm = torch.arange(bs, dtype=torch.long, device=d)

        wrapper = MultiLevelCascadeAttentionWrapper(
            num_levels=3,
            float_workspace_buffer=self.workspace_buffer,
            kv_layout="NHD",
            use_cuda_graph=True,
            qo_indptr_buf_arr=[qo_indptr_l0, qo_indptr_l1, qo_indptr_l2],
            paged_kv_indptr_buf_arr=[kv_indptr_l0, kv_indptr_l1, kv_indptr_l2],
            paged_kv_indices_buf_arr=[kv_indices_l0, kv_indices_l1, kv_indices_l2],
            paged_kv_last_page_len_buf_arr=[
                last_page_l0,
                last_page_l1,
                last_page_l2,
            ],
        )

        self._cg_cascade_wrappers[graph_key] = wrapper
        self._cg_cascade_buffers[graph_key] = {
            "qo_indptr_l0": qo_indptr_l0,
            "qo_indptr_l1": qo_indptr_l1,
            "qo_indptr_l2": qo_indptr_l2,
            "kv_indptr_l0": kv_indptr_l0,
            "kv_indptr_l1": kv_indptr_l1,
            "kv_indptr_l2": kv_indptr_l2,
            "kv_indices_l0": kv_indices_l0,
            "kv_indices_l1": kv_indices_l1,
            "kv_indices_l2": kv_indices_l2,
            "last_page_l0": last_page_l0,
            "last_page_l1": last_page_l1,
            "last_page_l2": last_page_l2,
            "perm": perm,
            "inv_perm": inv_perm,
        }

    def _fill_cg_cascade_plan(
        self,
        bs: int,
        graph_key: Any,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        common_prefix_tokens: int,
        rids: Optional[list[str]] = None,
        prefix_ref_rids: Optional[list[Optional[str]]] = None,
        shared_prefix_lens: Optional[list[Optional[int]]] = None,
        system_prefix_lens: Optional[list[Optional[int]]] = None,
        fallback_seq_lens: Optional[torch.Tensor] = None,
    ) -> bool:
        """Fill the per-bs cascade buffers in-place with the current step's
        metadata and call plan(). Returns False on failure (caller logs and
        falls through to parent's CG decode path).

        Buffers are written via ``.copy_()`` / index assignment so device
        addresses remain stable across replays (the captured graph reads
        from these same addresses).
        """
        self._last_cg_cascade_plan_failure = ""
        self._last_cg_cascade_plan_debug = {}
        wrapper = self._cg_cascade_wrappers.get(graph_key)
        bufs = self._cg_cascade_buffers.get(graph_key)
        if wrapper is None or bufs is None:
            return self._set_cg_cascade_plan_failure(
                f"missing wrapper/buffers for graph_key={graph_key}, bs={bs}"
            )

        meta = self._build_three_level_metadata(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens_cpu=seq_lens_cpu,
            rids=rids,
            prefix_ref_rids=prefix_ref_rids,
            shared_prefix_lens=shared_prefix_lens,
            system_prefix_lens=system_prefix_lens,
            fallback_common_prefix=common_prefix_tokens,
            fallback_seq_lens=fallback_seq_lens,
        )
        if meta is None:
            seq_lens_desc = None
            if seq_lens_cpu is not None:
                seq_lens_desc = seq_lens_cpu[:bs].cpu().tolist()
            elif fallback_seq_lens is not None:
                seq_lens_desc = fallback_seq_lens[:bs].detach().cpu().tolist()
            return self._set_cg_cascade_plan_failure(
                "metadata build returned None "
                f"(bs={bs}, common={common_prefix_tokens}, seq_lens={seq_lens_desc})"
            )

        system_prefix = meta["system_prefix"]
        groups = meta["groups"]
        seq_lens_list = meta["seq_lens"]
        per_req_shared = meta["per_req_shared"]
        perm_list = meta["perm"]
        inv_perm_list = meta["inv_perm"]
        rpi_list = req_pool_indices[:bs].cpu().tolist()

        bufs["kv_indptr_l0"].copy_(
            torch.tensor([0, system_prefix], dtype=torch.int32),
            non_blocking=True,
        )
        if system_prefix > bufs["kv_indices_l0"].numel():
            return self._set_cg_cascade_plan_failure(
                "level0 kv buffer too small "
                f"(system_prefix={system_prefix}, "
                f"capacity={bufs['kv_indices_l0'].numel()})"
            )
        if system_prefix > 0:
            shared_slots = self.req_to_token[int(rpi_list[0]), :system_prefix].to(
                torch.int32
            )
            bufs["kv_indices_l0"][:system_prefix].copy_(
                shared_slots, non_blocking=True
            )
            bufs["last_page_l0"].fill_(1)
        else:
            bufs["last_page_l0"].fill_(0)

        num_groups = len(groups)
        bufs["qo_indptr_l1"][: num_groups + 1].copy_(
            torch.tensor(meta["q_indptr_l1_cpu"], dtype=torch.int32),
            non_blocking=True,
        )
        if num_groups + 1 < bufs["qo_indptr_l1"].numel():
            bufs["qo_indptr_l1"][num_groups + 1 :].fill_(bs)

        kv_indptr_l1_cpu = [0]
        offset = 0
        bufs["last_page_l1"].fill_(0)
        for i, group in enumerate(groups):
            middle_len = max(0, group["shared_prefix"] - system_prefix)
            if middle_len > 0:
                if offset + middle_len > bufs["kv_indices_l1"].numel():
                    return self._set_cg_cascade_plan_failure(
                        "level1 kv buffer too small "
                        f"(group={i}, offset={offset}, middle_len={middle_len}, "
                        f"capacity={bufs['kv_indices_l1'].numel()})"
                    )
                anchor_rpi = int(rpi_list[group["anchor_idx"]])
                slots = self.req_to_token[
                    anchor_rpi, system_prefix : group["shared_prefix"]
                ].to(torch.int32)
                bufs["kv_indices_l1"][offset : offset + middle_len].copy_(
                    slots, non_blocking=True
                )
                bufs["last_page_l1"][i] = 1
                offset += middle_len
            kv_indptr_l1_cpu.append(offset)
        total_middle = offset
        bufs["kv_indptr_l1"][: num_groups + 1].copy_(
            torch.tensor(kv_indptr_l1_cpu, dtype=torch.int32),
            non_blocking=True,
        )
        if num_groups + 1 < bufs["kv_indptr_l1"].numel():
            bufs["kv_indptr_l1"][num_groups + 1 :].fill_(total_middle)

        bufs["perm"].copy_(
            torch.tensor(perm_list, dtype=torch.long), non_blocking=True
        )
        bufs["inv_perm"].copy_(
            torch.tensor(inv_perm_list, dtype=torch.long), non_blocking=True
        )

        unique_lens = [
            max(0, int(seq_lens_list[old_idx]) - per_req_shared[old_idx])
            for old_idx in perm_list
        ]
        kv_indptr_l1_cpu = [0]
        cum = 0
        for ul in unique_lens:
            cum += ul
            kv_indptr_l1_cpu.append(cum)
        total_unique = cum
        self._last_cg_cascade_plan_debug = {
            "layout_source": meta.get("layout_source", "unknown"),
            "system_prefix_tokens": int(system_prefix),
            "num_groups": int(num_groups),
            "group_shared_prefixes": [
                int(group["shared_prefix"]) for group in groups
            ],
            "group_member_counts": [
                len(group["members"]) for group in groups
            ],
            "per_req_shared": [int(x) for x in per_req_shared],
            "seq_lens": [int(x) for x in seq_lens_list],
            "total_middle": int(total_middle),
            "total_unique": int(total_unique),
            "perm": [int(x) for x in perm_list],
            "inv_perm": [int(x) for x in inv_perm_list],
        }
        bufs["kv_indptr_l2"].copy_(
            torch.tensor(kv_indptr_l1_cpu, dtype=torch.int32),
            non_blocking=True,
        )
        if total_unique > bufs["kv_indices_l2"].numel():
            return self._set_cg_cascade_plan_failure(
                "level2 kv buffer too small "
                f"(total_unique={total_unique}, "
                f"capacity={bufs['kv_indices_l2'].numel()}, "
                f"unique_lens={unique_lens})"
            )

        if total_unique > 0:
            offset = 0
            for new_i, old_i in enumerate(perm_list):
                ul = unique_lens[new_i]
                if ul == 0:
                    continue
                shared_i = per_req_shared[old_i]
                slots = self.req_to_token[
                    int(rpi_list[old_i]),
                    shared_i : shared_i + ul,
                ].to(torch.int32)
                bufs["kv_indices_l2"][offset : offset + ul].copy_(
                    slots, non_blocking=True
                )
                offset += ul

        bufs["last_page_l2"].fill_(1)

        try:
            # plan() runs host-side (it does .cpu() syncs internally to read
            # qo_indptr's last value etc.). It writes into the wrapper's
            # internal scheduler buffers, which are also fixed-shape since
            # the wrapper was built with use_cuda_graph=True.
            wrapper.plan(
                qo_indptr_arr=[
                    bufs["qo_indptr_l0"],
                    bufs["qo_indptr_l1"],
                    bufs["qo_indptr_l2"],
                ],
                paged_kv_indptr_arr=[
                    bufs["kv_indptr_l0"],
                    bufs["kv_indptr_l1"],
                    bufs["kv_indptr_l2"],
                ],
                paged_kv_indices_arr=[
                    bufs["kv_indices_l0"][: max(1, system_prefix)],
                    bufs["kv_indices_l1"][: max(1, total_middle)],
                    bufs["kv_indices_l2"][: max(1, total_unique)],
                ],
                paged_kv_last_page_len=[
                    bufs["last_page_l0"],
                    bufs["last_page_l1"],
                    bufs["last_page_l2"],
                ],
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads_local,
                head_dim=self.head_dim_local,
                page_size=self.cascade_page_size,
                causal=False,
                pos_encoding_mode="NONE",
                q_data_type=self.q_dtype,
                kv_data_type=self.kv_dtype,
            )
        except Exception as e:
            return self._set_cg_cascade_plan_failure(
                f"flashinfer wrapper.plan exception at bs={bs}: {type(e).__name__}: {e}"
            )
        return True

    # ------------------------------------------------------------------
    # Forward-metadata hooks
    # ------------------------------------------------------------------

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        # Always run the parent so non-cascade paths (extend, target verify,
        # draft extend, fallback decode) keep working.
        super().init_forward_metadata(forward_batch)

        # Reset both arms; CG hooks set their respective state below.
        self._in_cuda_graph = False
        self._cascade_plan = None
        self._cg_cascade_plan = None

        if not forward_batch.forward_mode.is_decode_or_idle():
            if self._dbg_enabled:
                self._dbg_skip_not_decode += 1
            return

        bs = int(forward_batch.req_pool_indices.shape[0])
        self._dbg_total_decode_steps += 1

        if bs < self.cascade_min_batch_size:
            self._dbg_skip_below_bs += 1
            return

        common = self._detect_common_prefix_tokens(forward_batch, bs)
        has_cascade_metadata = any(
            x is not None
            for x in (
                getattr(forward_batch, "cascade_shared_prefix_lens_cpu", None) or []
            )
        ) or any(
            x is not None
            for x in (getattr(forward_batch, "cascade_prefix_ref_rids", None) or [])
        ) or any(
            x is not None
            for x in (
                getattr(forward_batch, "cascade_system_prefix_lens_cpu", None) or []
            )
        )
        if (
            common < self.cascade_min_prefix_tokens
            and not has_cascade_metadata
            and not self._auto_detect_level1_enabled
            and not self._force_no_prefix_cascade
        ):
            self._dbg_skip_below_prefix += 1
            return

        plan = self._build_cascade_plan_args(forward_batch, bs, common)
        if plan is None:
            return
        self._cascade_plan = plan
        self._dbg_cascade_fired += 1
        if self._dbg_enabled:
            logger.info("Cascade fires: bs=%d, common_prefix_tokens=%d", bs, common)

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        # Parent allocates cuda_graph_kv_indices etc. for its decode wrappers.
        super().init_cuda_graph_state(max_bs, max_num_tokens, kv_indices_buf)

        # Size cascade buffers for the worst case in the captured set. We
        # don't know the full cuda_graph_bs list here -- only max_bs. Each
        # per-bs wrapper is allocated lazily in the capture hook.
        self._cg_max_bs = max_bs
        # Worst-case shared prefix length: cap at the model's context window
        # (cascade slot ids must lie within req_to_token's row width).
        self._cg_max_shared_pages = int(self.max_context_len)
        # Worst-case per-request tail length: same context window.
        self._cg_max_pages_per_req = int(self.max_context_len)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode,
        spec_info,
    ):
        super().init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_mode,
            spec_info,
        )

        self._in_cuda_graph = True
        self._cascade_plan = None
        self._cg_cascade_plan = None

        if forward_mode is not None and not forward_mode.is_decode_or_idle():
            return
        if self._cg_disabled:
            self._dbg_skip_in_cg += 1
            return
        if not is_flashinfer_available():
            return
        if bs < self.cascade_min_batch_size:
            self._dbg_skip_in_cg += 1
            return

        capture_batch = getattr(self, "_capture_forward_batch", None)
        variant_label = getattr(capture_batch, "cuda_graph_variant_label", None)
        if variant_label is None and capture_batch is not None:
            variant_label = self.get_cuda_graph_variant_label(capture_batch)
        graph_key = (int(bs), variant_label)

        if self._should_cg_no_prefix_fallback(variant_label):
            self._dbg_skip_in_cg += 1
            self._log_cg_no_prefix_fallback(
                "capture", bs, graph_key, variant_label
            )
            return

        self._allocate_cg_cascade_for_key(graph_key, bs)
        synth_seq_lens_cpu = (
            capture_batch.seq_lens_cpu
            if capture_batch is not None and capture_batch.seq_lens_cpu is not None
            else seq_lens.cpu()
        )
        capture_common = self._detect_common_prefix_from_rpi(
            bs, req_pool_indices, synth_seq_lens_cpu
        )
        ok = self._fill_cg_cascade_plan(
            bs,
            graph_key,
            req_pool_indices,
            synth_seq_lens_cpu,
            capture_common,
            rids=getattr(capture_batch, "rids", None),
            prefix_ref_rids=getattr(capture_batch, "cascade_prefix_ref_rids", None),
            shared_prefix_lens=getattr(
                capture_batch, "cascade_shared_prefix_lens_cpu", None
            ),
            system_prefix_lens=getattr(
                capture_batch, "cascade_system_prefix_lens_cpu", None
            ),
            fallback_seq_lens=seq_lens,
        )
        if ok:
            bufs = self._cg_cascade_buffers[graph_key]
            self._cg_cascade_plan = _CascadePlanState(
                common_prefix_tokens=capture_common,
                bs=bs,
                graph_key=graph_key,
                perm=bufs["perm"],
                inv_perm=bufs["inv_perm"],
            )
            self._log_cg_graph_key_decision(
                "capture", bs, graph_key, variant_label, capture_common
            )
        else:
            self._cg_cascade_wrappers.pop(graph_key, None)
            self._cg_cascade_buffers.pop(graph_key, None)
            if self._dbg_enabled:
                logger.warning(
                    "CG cascade capture-plan failed at bs=%d, graph_key=%s; "
                    "falling back to parent's per-request decode for this "
                    "graph. reason=%s",
                    bs,
                    graph_key,
                    self._last_cg_cascade_plan_failure,
                )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode,
        spec_info,
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        super().init_forward_metadata_replay_cuda_graph(
            bs,
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            encoder_lens,
            forward_mode,
            spec_info,
            seq_lens_cpu,
        )

        self._in_cuda_graph = True
        self._cascade_plan = None
        self._cg_cascade_plan = None

        if forward_mode is not None and not forward_mode.is_decode_or_idle():
            return
        if self._cg_disabled:
            self._dbg_skip_in_cg += 1
            return
        if not is_flashinfer_available():
            return
        replay_batch = getattr(self, "_replay_forward_batch", None)
        variant_label = getattr(replay_batch, "cuda_graph_variant_label", None)
        if variant_label is None and replay_batch is not None:
            variant_label = self.get_cuda_graph_variant_label(replay_batch)
        graph_key = (int(bs), variant_label)

        if self._should_cg_no_prefix_fallback(variant_label):
            self._dbg_skip_in_cg += 1
            self._log_cg_no_prefix_fallback("replay", bs, graph_key, variant_label)
            return

        if graph_key not in self._cg_cascade_wrappers:
            self._dbg_skip_in_cg += 1
            return

        common = self._detect_common_prefix_from_rpi(
            bs, req_pool_indices[:bs], seq_lens_cpu
        )
        ok = self._fill_cg_cascade_plan(
            bs,
            graph_key,
            req_pool_indices[:bs],
            seq_lens_cpu,
            common,
            rids=getattr(replay_batch, "rids", None),
            prefix_ref_rids=getattr(replay_batch, "cascade_prefix_ref_rids", None),
            shared_prefix_lens=getattr(
                replay_batch, "cascade_shared_prefix_lens_cpu", None
            ),
            system_prefix_lens=getattr(
                replay_batch, "cascade_system_prefix_lens_cpu", None
            ),
            fallback_seq_lens=seq_lens[:bs],
        )
        if not ok:
            if self._dbg_enabled:
                logger.warning(
                    "CG cascade replay-plan failed at bs=%d, common=%d "
                    "(captured graph may produce incorrect output for this "
                    "step). reason=%s",
                    bs,
                    common,
                    self._last_cg_cascade_plan_failure,
                )
            return

        bufs = self._cg_cascade_buffers[graph_key]
        self._cg_cascade_plan = _CascadePlanState(
            common_prefix_tokens=common,
            bs=bs,
            graph_key=graph_key,
            perm=bufs["perm"],
            inv_perm=bufs["inv_perm"],
        )
        self._log_cg_graph_key_decision(
            "replay", bs, graph_key, variant_label, common
        )
        self._dbg_cascade_run_cg += 1
        if common >= self.cascade_min_prefix_tokens:
            self._dbg_cascade_fired_cg += 1
            if self._dbg_enabled:
                logger.info(
                    "Cascade fires (CG): bs=%d, common_prefix_tokens=%d",
                    bs,
                    common,
                )
        else:
            self._dbg_skip_below_prefix_cg += 1
            if self._dbg_enabled:
                logger.info(
                    "Cascade ran (CG, below threshold): bs=%d, "
                    "common_prefix_tokens=%d",
                    bs,
                    common,
                )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        """CUDA-graph metadata hook (current out-of-graph API).

        Replaces the deprecated ``init_forward_metadata_{capture,replay}_cuda_graph``
        pair. The decode CUDA-graph runner calls this once per captured batch
        size with ``in_capture=True`` (before recording the graph), and again
        before every ``graph.replay()`` with ``in_capture=False``. Eager decode
        does not reach here -- it uses ``init_forward_metadata`` above.

          * Capture (``in_capture=True``): allocate the per-bs cascade wrapper,
            prime it with a synthetic plan, and arm ``_cg_cascade_plan`` so the
            ``forward_decode`` recorded into the graph invokes the cascade
            wrapper's ``run()`` for this bs.
          * Replay (``in_capture=False``): detect the actual shared prefix and
            refill the pre-allocated buffers in place so the recorded ``run()``
            reads the current step's metadata.
        """
        # Parent sets up its per-request decode wrappers for this bs -- our
        # fallback path, and required for non-decode capture modes.
        super().init_forward_metadata_out_graph(forward_batch, in_capture)

        self._in_cuda_graph = True
        self._cascade_plan = None
        self._cg_cascade_plan = None

        forward_mode = forward_batch.forward_mode
        if forward_mode is not None and not forward_mode.is_decode_or_idle():
            return
        if self._cg_disabled:
            self._dbg_skip_in_cg += 1
            return
        if not is_flashinfer_available():
            return

        bs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens_cpu = forward_batch.seq_lens_cpu
        variant_label = getattr(forward_batch, "cuda_graph_variant_label", None)
        if variant_label is None:
            variant_label = self.get_cuda_graph_variant_label(forward_batch)
        graph_key = (int(bs), variant_label)

        if bs < self.cascade_min_batch_size:
            # bs too small to ever fire cascade; the parent's per-request
            # decode handles this captured graph.
            self._dbg_skip_in_cg += 1
            return

        if self._should_cg_no_prefix_fallback(variant_label):
            self._dbg_skip_in_cg += 1
            self._log_cg_no_prefix_fallback(
                "capture" if in_capture else "replay",
                bs,
                graph_key,
                variant_label,
            )
            return

        if in_capture:
            # Allocate the per-bs cascade wrapper + buffers lazily, then prime
            # it with a synthetic plan (common_prefix=1 + bs unique tails) so
            # the captured run() has valid scheduler state. Capture-time slot
            # ids may point at zero-filled req_to_token rows; that is fine --
            # the captured kernel only records the launch, and replay overwrites
            # the buffers in place.
            self.prepare_cuda_graph_capture_forward_batch(forward_batch, variant_label)
            seq_lens_cpu = forward_batch.seq_lens_cpu
            self._allocate_cg_cascade_for_key(graph_key, bs)
            synth_seq_lens_cpu = (
                seq_lens_cpu
                if seq_lens_cpu is not None
                else forward_batch.seq_lens.cpu()
            )
            capture_common = self._detect_common_prefix_from_rpi(
                bs, req_pool_indices, synth_seq_lens_cpu
            )
            ok = self._fill_cg_cascade_plan(
                bs,
                graph_key,
                req_pool_indices,
                synth_seq_lens_cpu,
                capture_common,
                rids=forward_batch.rids,
                prefix_ref_rids=getattr(
                    forward_batch, "cascade_prefix_ref_rids", None
                ),
                shared_prefix_lens=getattr(
                    forward_batch, "cascade_shared_prefix_lens_cpu", None
                ),
                system_prefix_lens=getattr(
                    forward_batch, "cascade_system_prefix_lens_cpu", None
                ),
                fallback_seq_lens=forward_batch.seq_lens,
            )
            if ok:
                # Arm so forward_decode takes the cascade path during the
                # capture run; the captured graph then permanently invokes
                # wrapper.run(...) for this bs.
                bufs = self._cg_cascade_buffers[graph_key]
                self._cg_cascade_plan = _CascadePlanState(
                    common_prefix_tokens=capture_common,
                    bs=bs,
                    graph_key=graph_key,
                    perm=bufs["perm"],
                    inv_perm=bufs["inv_perm"],
                )
                self._log_cg_graph_key_decision(
                    "capture", bs, graph_key, variant_label, capture_common
                )
            else:
                # Capture-time plan failure: drop cascade for this bs and let
                # the captured graph use the parent's per-request decode.
                self._cg_cascade_wrappers.pop(graph_key, None)
                self._cg_cascade_buffers.pop(graph_key, None)
                if self._dbg_enabled:
                    logger.warning(
                        "CG cascade capture-plan failed at bs=%d, graph_key=%s; "
                        "falling back to parent's per-request decode for this "
                        "graph. reason=%s",
                        bs,
                        graph_key,
                        self._last_cg_cascade_plan_failure,
                    )
            return

        # ---- Replay: detect actual common prefix + refill buffers in place ----
        if graph_key not in self._cg_cascade_wrappers:
            # Cascade not captured for this bs (capture-plan failed, or
            # bs < min_batch_size at capture); the parent's path runs.
            self._dbg_skip_in_cg += 1
            return

        common = self._detect_common_prefix_from_rpi(bs, req_pool_indices, seq_lens_cpu)
        # Always plan (even below threshold): the captured graph for this bs has
        # cascade run() recorded, with no mid-graph fallback. Cascade with
        # common=0 is mathematically equivalent to per-request decode plus a
        # no-op level-0 launch, so always arm under CG.
        ok = self._fill_cg_cascade_plan(
            bs,
            graph_key,
            req_pool_indices,
            seq_lens_cpu,
            common,
            rids=forward_batch.rids,
            prefix_ref_rids=getattr(forward_batch, "cascade_prefix_ref_rids", None),
            shared_prefix_lens=getattr(
                forward_batch, "cascade_shared_prefix_lens_cpu", None
            ),
            system_prefix_lens=getattr(
                forward_batch, "cascade_system_prefix_lens_cpu", None
            ),
            fallback_seq_lens=forward_batch.seq_lens,
        )
        if not ok:
            if self._dbg_enabled:
                logger.warning(
                    "CG cascade replay-plan failed at bs=%d, common=%d "
                    "(captured graph may produce incorrect output for this "
                    "step). reason=%s",
                    bs,
                    common,
                    self._last_cg_cascade_plan_failure,
                )
            return

        bufs = self._cg_cascade_buffers[graph_key]
        self._cg_cascade_plan = _CascadePlanState(
            common_prefix_tokens=common,
            bs=bs,
            graph_key=graph_key,
            perm=bufs["perm"],
            inv_perm=bufs["inv_perm"],
        )
        self._log_cg_graph_key_decision(
            "replay", bs, graph_key, variant_label, common
        )
        self._dbg_cascade_run_cg += 1
        if common >= self.cascade_min_prefix_tokens:
            self._dbg_cascade_fired_cg += 1
            if self._dbg_enabled:
                logger.info(
                    "Cascade fires (CG): bs=%d, common_prefix_tokens=%d",
                    bs,
                    common,
                )
        else:
            self._dbg_skip_below_prefix_cg += 1
            if self._dbg_enabled:
                logger.info(
                    "Cascade ran (CG, below threshold): bs=%d, "
                    "common_prefix_tokens=%d",
                    bs,
                    common,
                )

    # ------------------------------------------------------------------
    # forward_decode (eager + CG dispatch)
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        # Resolve which cascade wrapper (if any) to invoke. Eager path uses
        # ``_cascade_decode_wrapper``; CG path uses the per-bs entry from
        # ``_cg_cascade_wrappers``. The selection is purely based on Python
        # state at call time; under CG capture/replay this resolves once at
        # capture (when ``_in_cuda_graph=True`` and ``_cg_cascade_plan`` is
        # set) and the captured graph then invokes wrapper.run() for that
        # specific wrapper instance every replay.
        if self._in_cuda_graph and self._cg_cascade_plan is not None:
            plan = self._cg_cascade_plan
            wrapper = self._cg_cascade_wrappers.get(self._cg_cascade_plan.graph_key)
        elif (not self._in_cuda_graph) and self._cascade_plan is not None:
            plan = self._cascade_plan
            wrapper = self._cascade_decode_wrapper
        else:
            plan = None
            wrapper = None

        if wrapper is None:
            return super().forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache
            )

        # Cascade arm: write KV-cache for current decode tokens, then run
        # the cascade wrapper.
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        if k is not None:
            assert v is not None
            if save_kv_cache:
                self.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        # KV pool returns (K, V), each [size+page_size, num_kv_heads, head_dim].
        # Cascade wrapper with page_size=1 expects [num_pages, 1,
        # num_kv_heads, head_dim] per K/V -- add a singleton page dim.
        k_buf, v_buf = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        kv_for_run = (k_buf.unsqueeze(1), v_buf.unsqueeze(1))

        q_3d = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        if plan is not None and plan.perm is not None:
            q_3d = q_3d.index_select(0, plan.perm)

        if not self._in_cuda_graph and not self._dbg_kernel_inputs_plan_only:
            self._log_eager_kernel_inputs(
                "Cascade eager run kernel inputs",
                {
                    "layer_id": layer.layer_id,
                    "bs": plan.bs if plan is not None else None,
                    "common_prefix_tokens": (
                        plan.common_prefix_tokens if plan is not None else None
                    ),
                    "save_kv_cache": save_kv_cache,
                    "cache_loc": self._tensor_debug_summary(cache_loc),
                    "q_original": self._tensor_debug_summary(q),
                    "k_current": self._tensor_debug_summary(k),
                    "v_current": self._tensor_debug_summary(v),
                    "q_for_run": self._tensor_debug_summary(q_3d),
                    "k_cache_for_run": self._tensor_debug_summary(kv_for_run[0]),
                    "v_cache_for_run": self._tensor_debug_summary(kv_for_run[1]),
                    "perm": self._tensor_debug_summary(
                        plan.perm if plan is not None else None
                    ),
                    "inv_perm": self._tensor_debug_summary(
                        plan.inv_perm if plan is not None else None
                    ),
                },
            )

        out = wrapper.run(q_3d, kv_for_run)
        if plan is not None and plan.inv_perm is not None:
            out = out.index_select(0, plan.inv_perm)
        return out.view(-1, layer.tp_q_head_num * layer.head_dim)

    # forward_extend is unchanged from the parent: cascade is decode-only
    # in this initial scope.
