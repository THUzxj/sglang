# 当前分支相对 SGLang v0.5.17 的改动说明

## 1. 对比范围与结论

本文描述 `codex/decode-log-server-time-v0517-ce` 相对
`origin/release/v0.5.17` 的提交态改动。基线与标签 `v0.5.17` 均指向
`29481685462732237d80d86076d6563e1f658102`；文档更新时 HEAD 为
`f8ffd6da67cc0b4cdf1d1df752ec72fd112a1e3d`。

- 分支领先 23 个提交（21 个普通提交、2 个 merge commit），没有落后。
- 差异涉及 42 个文件，约新增 5,915 行、删除 163 行。
- 核心是 Context Engineering Pair Scheduler，以及配套的 decode radix/HiCache、
  Mamba 状态管理和 PD cache-resume。
- 其他改动包括 metadata-only PD backend、时间与传输日志、指标、请求 metadata、
  DP rank 兼容和多项生命周期修复。
- 本文不把未提交文件算入分支差异。

## 2. 改动总览

| 主题 | 结果 |
| --- | --- |
| Pair Scheduler | main/foreground 优先；compact 只使用剩余 decode 预算，并按 `pair_key` 配对 |
| Compact pause | 支持 `retract`、`gpu_resident`、`radix_evictable` 三种模式 |
| PD cache resume | cache-paused compact 可通过 decode radix L1/L2/L3 恢复，并由 PD rebootstrap 补齐边界 |
| Full-KV-only match | Full+Mamba 的 PD decode 仅用 Full KV 限制 prefix match；Mamba 由 prefill 传入 |
| Decode HiCache | 修复 restore index、DMA 完成同步、KV release accounting 和 storage load-back |
| Hybrid Mamba offload | Decode offload 同时归档 Full KV 和有效 Mamba checkpoint |
| Metadata-only | 保留完整 PD control plane/aux metadata，跳过真实 KV payload copy |
| 接口与可观测性 | metadata/custom labels、DP rank alias、CE/时间/HiCache 日志与指标 |
| 稳定性 | 修复延迟 abort 误杀复用 RID、预分配失败回滚和 paused request 清理 |

## 3. 推荐部署

当前 Context Engineering 实验推荐：

```bash
--enable-pair-scheduler
--context-engineering-compact-pause-mode radix_evictable
--disaggregation-decode-enable-radix-cache
--enable-hierarchical-cache
--hicache-write-policy write_back
```

通常可以关闭 `--disaggregation-decode-enable-offload-kvcache`。原因是
`radix_evictable + write_back` 已能让暂停 compact 的 committed KV 从 request 私有状态转为
radix 所有：空间充足时留在 L1，显存紧张时写回 L2，恢复时走 L1 hit 或 L2/L3 load-back。

额外开启 decode offload 会增加一条“活跃 decode 按 stride 经临时 L2 staging 写 L3”的路径。
它适合持久化实验，但与 radix HiCache 可能重复写相同 page，不是 compact pause 的必要条件。

### 3.1 HiCache 组合

假设已开启 decode radix cache 和 hierarchical cache：

| Decode offload | Write policy | 活跃期/插入 | GPU 压力 | 恢复 |
| --- | --- | --- | --- | --- |
| 关闭 | `write_through` | radix 插入后异步复制 L2 | 有 L2 副本后可释放 L1 | L1 或 L2→L1 |
| 关闭 | `write_back` | radix KV 暂留 L1 | eviction 时 L1→L2 后释放 | L1 或 L2→L1 |
| 打开 | `write_through` | 专用 manager 按 stride 写 L3；radix 同时写 L2/L3 | radix 可释放 L1 | L1/L2/L3 分层恢复 |
| 打开 | `write_back` | 专用 manager 仍写 L3；radix 暂留 L1 | eviction 时 radix 写 L2/L3 | L1/L2/L3 分层恢复 |

Decode offload 要求 storage backend，所以专用 offload 实际是
L1→临时 L2 staging→L3，并非长期保留在专用 L2 pool。

## 4. Pair Scheduler

### 4.1 请求分类和 pair key

请求分为：

- `main`：前台 agent/main generation；
- `compact`：后台 context compaction；
- `foreground`：无 CE 标签的普通请求，优先级与 main 相同，但不能形成 pair。

分类来自 `Req.custom_labels`：

| 标签 | 类型 |
| --- | --- |
| `request_class=compact/compaction/context_engineering` 或 `call_kind=context_engineering` | compact |
| `request_class=main/agent/agent_run` 或 `call_kind=agent_run` | main |

Pair key 按 `pair_key`、`run_id`、`context_engineering_pair_key`、
`compact_pair_key`、label 中 `session_id`、`Req.session_id` 的顺序提取。

```json
{
  "text": "...",
  "custom_labels": {
    "request_class": "compact",
    "pair_key": "run-42",
    "allow_compact_drain": "false"
  }
}
```

若没有显式 `custom_labels`，原生 `/generate` 的 `metadata` 字典会转成字符串标签；
Session 派生请求也会继续携带标签。

### 4.2 Prefill

原 policy（priority、FCFS、LPM 等）排序后，再稳定分区为：

```text
[main + foreground，保持原相对顺序]
[compact，保持原相对顺序]
```

main/foreground 仍由普通 `PrefillAdder` 控制。compact 额外要求：

```text
active = running decode requests + 本轮已选 prefill requests
sum(max(kv_committed_len, seqlen)) + compact_tokens
    <= context_engineering_prefill_attention_token_budget
```

Prefill 不要求同 key main 已在运行，允许 compact-only prefill，也不为 waiting main 预留预算。
`compact_attention_cost_ratio` 当前不参与 prefill 计算。

### 4.3 Decode admission

每轮 decode 在普通 KV OOM 检查前执行：

1. main/foreground 用 `force=True` 全部保留，即使超过 CE budget。
2. compact 按 `pair_key` 建 FIFO 队列。
3. 按 batch 内 main 顺序，每个 main 尝试加入一个同 key compact。
4. compact 必须满足 compact request 数量上限和 attention-token budget。
5. 无匹配 main 的 compact 被暂停或 retract。

预算长度为：

```text
max(kv_committed_len, context_engineering_pause_committed_len, seqlen)
```

compact 成本为：

```text
ceil(attention_tokens * context_engineering_compact_attention_cost_ratio)
```

ratio 只影响 admission，不改变模型计算量。`context_engineering_compact_decode_max_batch_size`
只统计 compact request；main/foreground 不占用该额度。

可选参数 `--context-engineering-main-turn-decode-max-batch-size` 为 CE main turn 设置独立的
decode running 上限。超限 main 会 retract/requeue；unlabeled foreground 不计入该上限。参数未设置
时保持原行为，不单独限制 main turn 数量。

### 4.4 Compact-only、drain 和恢复

只剩 compact 时默认全部移出 decode。`allow_compact_drain=true` 的请求可在当前没有任何 main
时完成，但仍受 compact batch/token budget 限制。

统一恢复规则：非 compact 总能恢复；compact 只有在同 key main 正在运行，或允许 drain 且没有
任何 main 时，才能离开 paused/retracted 状态。

普通 OOM retraction 也 pair-aware：先撤 live pair 的 compact，重新检查内存；仍不足再撤同 pair
main，之后才处理其他 pair。

## 5. Compact Request Pause

### 5.1 三种模式

| 模式 | Request slot | GPU KV | Radix ownership | 恢复 |
| --- | --- | --- | --- | --- |
| `retract` | 释放 | 释放或保存私有 CPU copy | 通常无 | 普通 retract/rebootstrap |
| `gpu_resident` | 保留 | 保留 | 无变化 | 直接重建 decode batch |
| `radix_evictable` | 释放 | 初始可留 L1，之后可驱逐 | 有 | radix L1 hit 或 L2/L3 load-back |

### 5.2 GPU-resident

请求进入 `paused_compact_queue`，保留 `req_pool_idx`、page table、Mamba slot 和 decode progress。
恢复时直接构造 decode batch。若发生 KV 压力，优先处理与当前 main 同 pair 的 paused compact。
Unified HiCache 可把它插入 radix；PD decode 则保持 `Req.kv_cache_cpu` 私有 host-copy 契约。

### 5.3 Radix-evictable

暂停流程：

```text
release_kv_cache(is_insert=True)
  → committed KV 插入 radix，由 radix 接管 ownership
detach_gpu_state_for_cache_pause()
  → 释放 request slot/request-local GPU 绑定
  → 保留输出、grammar、logprob、sampling 等逻辑状态
  → 记录 context_engineering_pause_committed_len
```

在 `write_back` 下，KV 先作为可驱逐 radix L1；GPU 压力时普通 radix eviction 将其写入 L2。
恢复方式：

- Unified：回到 waiting queue，经正常 prefix match/admission；
- PD decode：进入 cache-resume prealloc，以 `origin_input_ids + output_ids` 做本地 L1/L2/L3
  匹配；未命中后缀和 handoff boundary 由原 prefill worker rebootstrap。

PD 会保存已经输出的 boundary token，并覆盖重算时的新 handoff token，防止重复输出；原历史
logprob 也不会再次提交。恢复时重置 admission 时间戳，避免导出负 queue latency。

`radix_evictable` 要求开启 pair scheduler 和 hierarchical cache；PD decode 还要求开启 decode
radix cache。Abort、reset、idle 和 `/v1/loads` 均纳入 paused queue。

## 6. PD Disaggregation 与 Decode Prefix Restore

### 6.1 分层 prefix reuse

Decode preallocation 先本地 match：

```text
L1 hit       → 直接写 request page table
L2 host hit  → 预留 GPU slots，异步 H2D
L3 hit       → L3→L2→L1
未命中 suffix → 只给 delta 分配目标 page，交给 prefill
```

`decode_prefix_len = L1 + L2 + L3` 发给 prefill，使其只处理之后的 delta。Pending load-back token
从可分配预算扣除，防止多个恢复请求超额承诺 GPU slots。

### 6.2 Restore index 修复

`init_load_back()` 返回的整段 radix path 可能与已有 L1 prefix 重叠，不能直接写 page table。
现在 load-back 安装 mapping 后用冻结 token key 再 match，精确截取
`[original_l1_prefix_len, decode_prefix_len)`；长度完全一致才提交，否则 restore failed。

### 6.3 DMA 完成 gate

请求进入 waiting/running 必须同时满足：

```text
PD metadata/KV transfer success
AND
本地 HiCache H2D device event 完成
```

`is_load_back_event_done()` 查询 event 并推进 ack/lock 生命周期；提交新 load-back 前检查下一个
ring counter slot，避免覆盖仍在执行的 event。

### 6.4 Full-KV-only matching

对 Full+Mamba 且 PD 会传 `StateType.MAMBA` 的模型，decode prefix lookup 只要求 Full KV 命中：

- Full KV 从 decode radix L1/L2/L3 复用；
- Mamba post-prefill state 由本次 prefill 传入；
- 缓存 Mamba 缺失/较短不会截短 Full KV hit；
- Full+SWA+Mamba 不启用，因为 SWA 另有 request-local reuse 约束。

Unified tree 新增 `MatchPrefixParams.match_full_kv_only`，只运行 Full component validator/finalizer，
并返回 `full_kv_hit_length`。

## 7. Mamba 与 Hybrid Decode Offload

Decode offload 现在支持：

```text
HostPoolGroup
├── KV anchor：Full-Attention MHA/MLA KV
└── MAMBA side pool：request-level checkpoint
```

按 device pool 大小拆分 `hicache_size`，使用 `HybridCacheController`。Mamba 采用
`TRAILING_PAGES`：extra-buffer 仅保存与 chunk end 对齐的 ping-pong keep slot；lazy checkpoint
无效时跳过；无 extra-buffer 时仅在请求完成且状态稳定时保存 active slot；storage key 为当前
KV chunk 最后一个 page hash。没有有效 checkpoint 时只归档 Full KV。

Decode 分配前会计算 active slot 和 tracking slots 的完整 Mamba 需求，不足时先驱逐 radix 中的
Mamba state。`MambaSlotAllocator` 增加 group allocation 和 `SGLANG_DEBUG_MEMORY_POOL` 事件日志。

KV 生命周期修复还包括：

- 活跃 decode 中不再过早释放仍被 attention 使用的 prefill KV；
- 完成后统一释放，并尊重 `cache_protected_len`；
- 用带逻辑起点的 `free_segment()`，清理 speculative over-allocation 和 Mamba slot；
- clamp `protected_size_`，限制 `evictable_size` 不超过实际 allocation；
- admission 前 flush write-through ack；
- storage hit-query 联合验证 Mamba trailing checkpoint。

## 8. Metadata-only PD Backend

新增 `metadata_only`，语义是“真实 control plane，无 KV data plane”：

```text
Decode ── room、目标 pages、aux/state indices、decode_prefix_len ──> Prefill
Decode <──────── 完整 first-token/aux metadata ─────────────────── Prefill
Decode <────────────── 不传真实 KV tensor payload ─────────────── Prefill
```

保留 bootstrap、rank registration、TP/CP/PP mapping、heartbeat、decode preallocation、chunked
prefill、abort，以及 output id、cached-token、logprob、sampling mask、top-k、hidden state、DSA
top-k 和 `bootstrap_room` gate。省略 NIXL/RDMA agent、memory registration 和 KV/state copy。

Prefill/Decode manager 注入 `MetadataBuffers`；最后一个 chunk 用 ZMQ 回传完整 aux slot，decode
写入本地 slot 后才把 room 标记成功。该 backend 用于测 control-plane/batching 开销，不能验证
真实 KV 内容或数据面带宽。

## 9. Metadata、DP Routing 与生命周期

- `GenerateReqInput` 新增 `metadata`；`TokenizedGenerateReqInput.custom_labels` 将其传播到
  scheduler `Req` 和 Session。
- 原生 `/generate` 接受旧 `data_parallel_rank`，警告后迁移到 `routed_dp_rank`；两者并存时以
  新字段为准。
- Pair Scheduler 是单 scheduler/DP rank 内机制，不自动提供跨 worker pair affinity；上游需通过
  router 或 `routed_dp_rank` 把一对请求送到相同 worker。
- 延迟 background abort 现在同时验证 RID 仍存在且 `state.obj` 是创建任务的原 request object，
  防止 RID 被复用后旧任务误杀新请求。
- Abort 扫描扩展到 paused、prealloc、transfer、retracted、HiCache prefetch 和 restore lock；
  cache-paused 与 GPU-resident 分别按 ownership 清理，避免 double free/遗留锁。

## 10. 日志与指标

Pair Scheduler 日志增加 `#ce-main`、`#ce-compact`、`#ce-paired-compact`，以及结构化事件：

```text
context_engineering_batch
context_engineering_park
context_engineering_retract
context_engineering_resume
```

事件含 stage、forward iteration、TP/PP/DP rank、RID、kind、pair key、seqlen、committed length、
配对状态和 reason；例行日志仅 primary TP stats rank 输出。

新增 Prometheus gauge：

```text
sglang:prefill_input_throughput_window_s
sglang:prefill_batch_start_end_s
```

Decode 日志增加 wall-clock server time 和毫秒级 log interval。Paused queue 计入 load snapshot 的
waiting/paused 和 idle 判定，但不计为待 prefill token。只有已注册 custom label key 才进入
Prometheus，避免动态 schema 和无界基数。

HiCache DEBUG 日志覆盖 L1→L2 write-through/write-back 和 L2→L1 load-back，记录 node、token、
估算 bytes、elapsed time 和有效 GB/s。

## 11. 可靠性修复

- Decode 实际 preallocation 在估算后失败时，释放 request slot、清空 KV、归还 prefix lock，
  请求留待下一轮重试。
- Priority 排序移到所有 index-based queue bookkeeping 之前。
- Restore commit 前严格验证区间长度。
- Metadata transfer success 与本地 load-back completion 正确 gate。
- Hybrid SWA/SSM decode radix 的启动限制从 hard error 改为 experimental warning；这不代表所有
  model/backend 组合都已完成正确性验证。

## 12. 测试覆盖

新增/扩展测试覆盖：Pair 排序和 admission、compact-only/drain、attention ratio、两种 cache pause、
KV pressure、Hybrid Mamba offload、精确 restore index、load-back event gate、Full-KV-only match、
preallocation rollback/lock-ref、DP alias，以及延迟 abort/RID reuse。

```text
test/srt/test_pair_scheduler.py
test/srt/test_pair_scheduler_kv_pressure.py
test/registered/unit/disaggregation/test_decode_hicache_restore.py
test/registered/unit/mem_cache/test_hicache_load_back_event_sync.py
test/registered/unit/mem_cache/test_decode_kvcache_offload_hybrid.py
test/registered/unit/mem_cache/test_decode_radix_lock_ref.py
test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py
test/registered/unit/managers/test_io_struct.py
test/registered/unit/managers/test_tokenizer_manager_rid_cleanup.py
```

## 13. 当前边界与注意点

- 没有跨 DP worker 自动 pair affinity/pair-density balancing。
- 没有独立 pair-level 全局 FCFS/tool-call 状态机；依赖请求标签和每轮 batch admission。
- Prefill 只做 main-first/剩余容量限制，不做严格一对一 pair admission。
- ratio 的帮助文字提到 prefill/decode，但当前只在 decode 生效。
- `select_decode_keep_indices()` 的 `waiting_queue` 参数未使用；prefill helper 的 waiting queue 和
  ratio 参数也未使用。
- `select_compact_indices_after_paired_main_finished()` 有实现和单测，但生产路径主要依靠下一轮
  compact-only selection，没有直接调用它。
- `radix_evictable` 并非零成本：L2/L3 hit 需要 DMA，PD 还需 rebootstrap/aux handshake。
- Full-KV-only 不覆盖 Full+SWA+Mamba。
- Metadata-only 不能验证 KV data plane。

## 14. 提交演进

```text
c63f1fdda4  Log decode batch server time
0507fbcaa1  Add initial context engineering scheduler
c62558f756  Bypass not implemented error
b0350bfc73  Add mamba hicache on decode
dc1cb64e04  Add metadata-only KV transfer backend
228e669d40  Update decode batch server time unit
f87a9a5c17  Accept deprecated data_parallel_rank alias on native /generate
f65ce7374f  Fix decode Mamba prealloc eviction
64de14e6aa  Fix decode HiCache KV release accounting
216b6cbb1b  Implement CE pair scheduler admission
1ad8cbb498  Add prefill timing metrics and sanitize custom labels
fd73729f5d  Rename scheduler to pair scheduler
36e4cfda48  Limit transfer logs to primary TP ranks
d5fd528f6b  Add GPU-resident compact pause queue
f8f1b21204  Add HiCache KV transfer debug logs
86452ef5c7  Fix decode HiCache restore index mapping
8a8c8cbb46  Guard delayed abort request lifecycles
2300eb3284  Restore cache-paused requests on PD decode
89290ed0f4  Add radix-evictable compact pause mode
8c754d7bbe  Synchronize HiCache load-back completion
f8ffd6da67  Enable Full-KV-only PD prefix matching
```

Merge commits：`36d1114eee`、`346934b2f1`。

## 15. 文件级索引

### Pair Scheduler

```text
python/sglang/srt/managers/pair_scheduler.py
python/sglang/srt/managers/scheduler.py
python/sglang/srt/managers/schedule_batch.py
python/sglang/srt/managers/schedule_policy.py
python/sglang/srt/managers/scheduler_pp_mixin.py
python/sglang/srt/managers/scheduler_components/load_inquirer.py
python/sglang/srt/managers/scheduler_components/metrics_reporter.py
python/sglang/srt/server_args.py
```

### PD、HiCache 与 Mamba

```text
python/sglang/srt/disaggregation/decode.py
python/sglang/srt/disaggregation/decode_hicache_mixin.py
python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py
python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py
python/sglang/srt/disaggregation/prefill.py
python/sglang/srt/mem_cache/base_prefix_cache.py
python/sglang/srt/mem_cache/allocator/mamba.py
python/sglang/srt/mem_cache/memory_pool.py
python/sglang/srt/mem_cache/kv_cache_builder.py
python/sglang/srt/mem_cache/hiradix_cache.py
python/sglang/srt/mem_cache/hi_mamba_radix_cache.py
python/sglang/srt/mem_cache/unified_radix_cache.py
python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py
python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py
python/sglang/srt/managers/cache_controller.py
```

### Metadata-only、接口与指标

```text
python/sglang/srt/disaggregation/metadata_only/__init__.py
python/sglang/srt/disaggregation/metadata_only/conn.py
python/sglang/srt/disaggregation/utils.py
python/sglang/srt/managers/io_struct.py
python/sglang/srt/managers/tokenizer_manager.py
python/sglang/srt/session/session_controller.py
python/sglang/srt/observability/metrics_collector.py
python/sglang/srt/observability/req_time_stats.py
```

设计说明：`metadata_only_kv_transfer.md` 和
`python/sglang/srt/disaggregation/hybrid_decode_kvcache_offload_design.md`。
