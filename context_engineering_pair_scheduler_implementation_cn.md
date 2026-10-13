# SGLang v0.5.17 Pre-compaction Pair Scheduler 实现审计与落地方案

本文档面向 `references/sglang_dev`，说明当前 Context Engineering（CE）调度代码为什么尚不能运行、它与论文 `paper_draft.md` 中 **Pre-compaction pair scheduling** 的差距，以及在当前 SGLang 分支上逐步实现、验证论文方案的具体方法。

审计基线：

- 当前 `HEAD`：`346934b2f1ab513f34e81231665f09924c6931c4`，其中合入了 `b0350bfc73 Add mamba hicache on decode`。
- CE 初始提交：`0507fbcaa1`。
- Qwen3.5 decode HiCache 初始提交：`b0350bfc73`（通过 merge commit `346934b2f1ab5` 进入当前分支）。
- 论文依据：`serving_system/paper_draft.md` 的 `Pre-compaction pair scheduling`。
- 详细语义依据：`serving_system/parallel_compaction_v1_dev_doc_cn.md` 第 9.3 节。
- 审计日期：2026-09-03。

## 1. 结论

当前提交只能称为 **未接通的 single-worker main-first prototype**，不能作为论文 pair scheduler 的实现，也不能用于生成论文的 pair-scheduler 实验结果。

原因分为四层：

1. **当前分支本身不可正常运行。** CE 提交误删了多个既有方法，又无条件调用了不存在的方法。普通 prefill/decode、decode OOM 和 PD rebootstrap 都存在确定的运行时错误。
2. **新增 helper 没有接入调度主循环。** prefill 排序、compact admission、decode budget retract 都只有函数或方法定义，没有有效调用点。
3. **论文核心机制尚未实现。** 当前没有 pair-FCFS、pair 状态机、tool-call 同步挂起/恢复、compact-only drain 例外、pair affinity、跨 worker pair-density balance，也没有把 shared-prefix cascade attention 接到 serving decode 路径。
4. **Qwen3.5 HiCache 只完成 archive write 原型。** 当前仍有 finished release 的确定异常、L3 hit-query 未实现、双 controller 冲突和 hybrid snapshot 恢复缺口，不能据此宣称支持 decode retract/resume。

必须先恢复 SGLang 基线，再分阶段实现 scheduler、生命周期、路由和 kernel。不要在当前提交上直接开始性能实验。

## 2. 论文语义边界

论文中的 pair scheduler 不是简单的 main/compact priority，也不是“main 全部保留，compact 使用剩余资源”。目标语义如下。

### 2.1 Prefill worker

- prefill 不组成执行 pair。
- foreground/main 位于高优先级队列，compact 位于低优先级队列。
- 每个队列内部按 agent/pair 到达顺序 FCFS。
- 每轮先消费 main queue，再消费 compact queue；compact 使用剩余的真实 prefill token、slot 和 KV 资源。
- compact 的 chunked prefill 尚未完成时，如果 main 到达，应在下一个 chunk 边界停下 compact，先调度 main。
- compact-only prefill 合法。

### 2.2 Decode worker

- 排队和 admission 单位是 `PairState`，不是两个临时独立请求。
- pair 按 agent 到达顺序 FCFS。
- 同一 pair 的 main 和 compact 同时 runnable 时，每个 decode iteration 必须共同进入 batch。
- main-only 只在 compact 尚未触发或已经完成时合法。
- compact-only 只在 main 达到 hard limit、没有 completed checkpoint 可用于 Stage 1 switch 时合法；该状态记为 `COMPACT_DRAIN`。
- tool call、资源 preemption 和需要释放 KV 的 OOM 操作必须保持 pair 一致性。

### 2.3 Worker 间调度

- affinity 是硬约束：同一 `pair_key` 的所有 main turns、compact logical requests 和 resume attempts 固定到同一 decode worker。
- density balance 是 affinity 之后的新 pair placement：在基本 KV/queue/decode load 可接受的候选 worker 中，平衡 paired compact decode work 占全部 decode work 的比例。

### 2.4 Prefix-share kernel

pair 同 batch 只是启用优化的必要条件，不是优化本身。论文声称的“compact 计算近似免费”要求 serving decode 真正执行 prefix-aware/cascade attention，且 kernel 使用了经验证相同的 shared KV pages。

当前 `flashinfer_backend.py::forward_decode()` 仍调用普通 `BatchDecodeWithPagedKVCacheWrapper.forward()`。仓库中的 `benchmark/cascade_kernels` 已有 `MultiLevelCascadeAttentionWrapper` microbenchmark，但没有接入 SGLang serving。因此 scheduler 实验与 kernel 实验必须分开标记，直到 serving kernel 路径完成。

## 3. 当前代码地图

已有代码主要包括：

| 位置 | 当前作用 | 状态 |
| --- | --- | --- |
| `python/sglang/srt/managers/io_struct.py` | `/generate` 接受 `metadata`，tokenized request 携带 `custom_labels` | 已接通，但仍是非类型化兼容层 |
| `python/sglang/srt/managers/tokenizer_manager.py` | `metadata` 在无显式 `custom_labels` 时被字符串化并转成 labels | 已接通 |
| `python/sglang/srt/managers/schedule_batch.py::Req` | 推导 main/compact 和 pair key | 已接通，但 key 规则错误且字段不足 |
| `pair_scheduler.py` | prefill stable partition、decode keep 选择、统计 helper | 纯函数存在，核心调度未接通 |
| `scheduler.py` | 定义 CE 开关和两个 retract/order wrapper | wrapper 存在，但无有效调用或引用缺失 |
| `ScheduleBatch.retract_decode()` | 尝试 OOM 时原子 retract 一对 main/compact | 部分接入，但实现不完整 |
| `decode_kvcache_offload_manager.py` | decode 增量 KV + Mamba checkpoint 写入 L3 | write prototype；不是 request resume store |
| `unified_radix_cache.py` | hybrid FULL/MAMBA radix 与 HiCache | L1/L2 基础存在；L3 hit query 仍未实现 |
| `metrics_reporter.py` | batch 中 main/compact/paired 数量和结构化日志 | 已接入统计点，但主路径会先崩溃 |
| `server_args.py` | pair scheduler 和 budget 参数 | 已声明，无完整行为和参数校验 |
| `data_parallel_controller.py` | 支持显式 `routed_dp_rank` | 可作为 affinity 基础，尚无 pair-density policy |

`bootstrap_pair_key` 当前只在 `io_struct.py` 中声明、归一化和复制，没有调度或路由 consumer，不能把它当作已实现的 pair affinity。

## 4. 已确认问题

### 4.1 P0：阻止运行或破坏原有功能

#### P0-1：普通 forward result 会调用不存在的方法

`scheduler.py::process_batch_result()` 当前无条件执行：

```python
if batch.forward_mode.is_decode():
    self.batch_result_processor.process_batch_result_decode(batch, result)
    self._retract_compact_after_paired_main_finished(batch)

self._record_step_counters(batch, result)
```

但 `Scheduler` 中没有 `_retract_compact_after_paired_main_finished()`，也没有 `_record_step_counters()`。后者原本存在，被 `0507fbcaa1` 删除。

影响：

- 第一个 prefill result 会在 `_record_step_counters` 处失败。
- 第一个 decode result 会更早在 `_retract_compact_after_paired_main_finished` 处失败。
- 两个调用都没有受 `enable_pair_scheduler` 保护，关闭 CE 也无法规避。
- `_record_step_counters` 被删除还会破坏 `/v1/loads` 使用的 prefill busy/token counter 和 decode step moments。

P0 修复：从 `0507fbcaa1^` 原样恢复 `_record_step_counters()`；先删除不存在的 CE finish 调用，后续由 PairCoordinator 在正确的调度边界实现 finish event，不能在这里补一个直接修改 batch 的临时方法。

#### P0-2：decode OOM 排序方法被删除

`ScheduleBatch.retract_decode()` 仍调用：

```python
self._get_decode_retraction_order(self.reqs, server_args)
```

但 `0507fbcaa1` 删除了 `_get_decode_retraction_order()`。任何触发 decode KV OOM/retraction 的请求都会失败。

P0 修复：先从父提交完整恢复该方法和原有测试。pair-aware OOM 应在后续提交中基于“retraction group”扩展，而不是删除原排序逻辑。

#### P0-3：PD true-retraction payload 方法被删除

`python/sglang/srt/disaggregation/decode.py` 仍调用 `req.build_rebootstrap_payload()`，相关单测也直接验证它，但 `Req.build_rebootstrap_payload()` 已被删除。

P0 修复：从父提交恢复该方法，并在返回 payload 中增加：

```python
"custom_labels": dict(self.custom_labels),
```

这样 rebootstrap 后的 physical attempt 才不会丢失 `pair_key`、logical compact identity 和 generation 信息。

#### P0-4：新增方法引用未导入符号

`scheduler.py` 中：

- `_order_context_engineering_waiting_queue()` 调用未导入的 `order_prefill_waiting_queue`。
- `_retract_compact_decode_over_budget()` 调用未导入的 `select_decode_keep_indices`。

当前因方法没有被调用而没有立刻触发；一旦把 helper 接到主循环，就会变成 `NameError`。

#### P0-5：日志方法不存在

- `_retract_compact_decode_over_budget()` 调用不存在的 `self._log_context_engineering_retract()`。
- `scheduler_pp_mixin.py::process_retract_queue()` 调用不存在的 `self._log_context_engineering_resume()`。

PP disaggregated decode 一旦处理 retracted queue 就会失败。应在 `MetricsReporter` 中实现统一的 `log_context_engineering_transition(...)`，Scheduler 只转发事件；不要保留多个未定义的私有日志入口。

#### P0-6：Ascend 初始化 import 被误删

`scheduler.py` 仍调用 `maybe_create_ascend_config_store()`，但 import 被 `0507fbcaa1` 删除。需要恢复：

```python
from sglang.srt.managers.disagg_service import maybe_create_ascend_config_store
```

#### P0-7：两个非 CE 参数被错误替换

`_get_new_batch_prefill_raw(..., running_batch)` 内部两处把传入参数换成了 `self.running_batch`：

```python
self.policy.calc_priority(self.waiting_queue, self.running_batch)
running_bs = len(self.running_batch.reqs)
```

在普通单 microbatch 上二者经常恰好相同，但 PP/microbatch 路径依赖传入的 `running_batch`，不能混用全局临时字段。两处都应恢复为参数。

#### P0-8：已知不兼容配置被无条件放行

`kv_cache_builder.py` 被改成允许 `disaggregation_decode_enable_radix_cache` 与 hybrid SWA/SSM 组合继续启动，而原实现明确抛出 `ValueError`。Qwen3.5 的目标部署确实需要逐步支持 hybrid SSM，但当前代码只补了部分接口，仍会进入确定的异常或未验证路径。

P0 修复：不要简单恢复“一律拒绝”，也不能保留“一律 warning 后放行”。应增加 capability gate，只允许已经验证的 Qwen3.5 基线组合；L3 storage、speculative decode、unified memory、PP/overlap 等尚未闭环的组合继续 fail-fast。具体矩阵和实现步骤见第 17 节。

### 4.2 P1：代码存在但调度语义没有生效

#### P1-1：三个核心 helper 没有主循环调用点

当前搜索结果显示：

- `_order_context_engineering_waiting_queue()` 只有定义。
- `should_try_prefill_request()` 只有定义和 helper 单测。
- `_retract_compact_decode_over_budget()` 只有定义。

因此现在真正生效的 CE 行为只有 request 分类、部分日志字段和 OOM pair 尝试；正常 prefill/decode admission 没有改变。

#### P1-2：compact admission 失败后的 `continue` 不是双队列调度

当前在 `PrefillAdder.add_one_req()` 返回非 `CONTINUE` 后，仅当失败请求是 compact 时继续扫描 waiting queue。它不能建立 `Q_main/Q_compact`，也没有对 compact chunk 做 preemption；反而可能在请求已经执行 `init_next_round_input()`、LoRA 状态更新或部分 cache bookkeeping 后改变原有退出语义。

P0 阶段应先撤掉这处特殊分支。正式 prefill policy 应在排序和 admission 前显式完成 queue class 处理。

#### P1-3：chunked compact 会优先于后来到达的 main

`_get_new_batch_prefill_raw()` 在扫描 waiting queue 前无条件调用：

```python
self.chunked_req = adder.add_chunked_req(self.chunked_req)
```

所以正在执行的 compact chunk 在下一个调度周期仍先被加入，违反论文“main 到达后在下一 chunk 边界抢占 compact”的要求。

#### P1-4：当前 finish hook 位于错误对象和错误时机

在 overlap scheduler 中，`process_batch_result()` 收到的可能是 `batch.copy()` 对应的历史 batch。即使补出 `_retract_compact_after_paired_main_finished()`，直接在该对象上 release/filter 也不能可靠更新下一轮使用的 live `running_batch`。

正确做法是：result processor 只产生 pair lifecycle event；实际 batch mutation 在下一次 scheduling boundary、针对 live running batch 执行。初版若不能证明 overlap 正确，应让 `strict_pair` 模式 fail-fast 要求 `--disable-overlap-schedule`。

#### P1-5：budget retract 即使接入也会丢请求

`_retract_compact_decode_over_budget()` release/filter 后只返回 `retracted_reqs`。如果调用者忽略返回值，这些请求不会进入普通 waiting queue 或 PD `DecodePreallocQueue.retracted_queue`，逻辑请求会消失。

所有 retract 入口必须统一经过：

```python
for req in retracted_reqs:
    self._add_request_to_queue(req, is_retracted=True)
```

同时要清理 `batch_is_full`、更新 retraction counter，并记录 pair transition。

#### P1-6：OOM 的“原子 pair retract”仍有边界错误

当前实现从 request index 列表 pop 一个请求，再尝试移除其 counterpart。问题包括：

- 两请求 pair 可能一次被删到 `sorted_indices=[]`，后续 last-request abort 分支仍可能 `pop()` 空列表。
- “至少保留一个 request”不等于“至少保留一个 scheduling group”。
- 同 pair 多个 main 或多个 compact 没有被拒绝，counterpart 选择不确定。
- `remaining_req_count` 仍按 index list 近似，pair 删除后 bookkeeping 口径不清楚。
- pair 的两侧是否都进入可恢复队列，没有在 `ScheduleBatch` 层保证。

应改为 `_get_decode_retraction_groups()` 返回 `List[List[int]]`，普通请求是一元素 group，runnable pair 是二元素 group；OOM 循环按 group pop、release、requeue。

#### P1-7：pair identity 不稳定

当前 pair key 读取顺序为：

```text
context_engineering_pair_key
compact_pair_key
pair_key
session_id
run_id
```

这会让兼容字段覆盖 canonical `pair_key`，并优先使用可能跨并发 run 重复的 `session_id`。当前 serving runtime 还曾把 generation 拼入 pair key，使一次 context switch 后同一 agent 被拆成新 pair。

论文对齐的定义必须是：

```text
pair_key = run_id                         # 跨 generation 稳定
shared_prefix_group_id = run_id:generation-N
pair_arrival_sequence = agent arrival sequence
```

旧字段只在 API 边界兼容读取，解析后立刻归一为一个 canonical field。

#### P1-8：compact-only decode 被无条件禁止

`select_decode_keep_indices()` 对所有 compact-only batch 返回空列表，且传入的 `waiting_queue` 实际没有使用。这与论文 hard-limit fallback 冲突。

必须由显式 pair state `COMPACT_DRAIN` 或 `allow_compact_drain=true` 放行。scheduler 不应根据“当前没看见 main”自行推断 hard-limit 或 checkpoint 状态。

#### P1-9：没有 PairState，不满足 pair-FCFS

当前 selector 先强制保留所有 main/foreground，再按 main 顺序尝试放 compact。这是 `main_first_degraded_pair`，可能在 budget 边界拆开同时 runnable 的 pair，也无法保持原 agent arrival order。

论文模式必须以 pair 为 admission 单位，并显式拒绝同一 pair 同时存在多个 active compact side。

#### P1-10：配置含义不一致

- `context_engineering_compact_decode_max_batch_size` 当前只限制 compact physical requests，论文按 pairs 计数。
- `context_engineering_compact_attention_cost_ratio` 的 help 声称用于 prefill/decode，但 prefill helper 故意忽略它。
- CE 参数没有非负值、ratio 范围和组合合法性校验。
- 当前 prefill helper 用整个 `seqlen/kv_committed_len`，而论文和 SGLang `PrefillAdder` 的关键约束是本轮实际 uncached/chunk tokens。

建议新增 `context_engineering_max_batch_pairs`；prefill 直接复用 `PrefillAdder` 的真实 token budget，不再引入含义重复的 CE attention budget。cost ratio 只保留给 degraded-pair 消融或经过校准的 pair cost model。

### 4.3 P2：性能机制、路由和可观测性缺失

- 没有 tool-call pair control API。
- 现有 `/pause_generation` 是暂停整个 engine，不能用于单个 pair。
- PD decode 的 `resume_retracted_reqs()` 逐请求恢复，不保证 pair all-or-none capacity check。
- 没有 `pair_key -> dp_rank` affinity map。
- `LoadSnapshot` 没有 CE pair/density/window counters。
- 没有 pair-density placement。
- 没有 serving cascade attention。
- `context_engineering_batch` 的 `paired` 只表示 batch 中出现相同 key，不能证明两侧此前同时 runnable，也不能证明实际共享 prefix。
- helper 单测通过 direct file import，不能发现 Scheduler 主模块中的未定义方法、错误 import 和未接线问题。

## 5. 目标数据模型

不要让 scheduler 在每轮从任意字符串 labels 反复猜测状态。API 边界可保留兼容字段，但 tokenization 后必须形成类型化元数据。

建议新增：

```python
class ContextEngineeringRequestInfo:
    request_class: Literal["main", "compact"]
    pair_key: str
    pair_arrival_sequence: int
    shared_prefix_group_id: str
    logical_compaction_id: Optional[str]
    attempt_index: int
    switch_epoch: Optional[int]
    allow_compact_drain: bool
    suspend_pair_on_finish: bool
    expected_main_decode_tokens: Optional[int]
    expected_compact_decode_tokens: Optional[int]
```

字段流向：

```text
GenerateReqInput
  -> TokenizerManager validate/normalize
  -> TokenizedGenerateReqInput.context_engineering
  -> DataParallelController affinity/density routing
  -> Scheduler.handle_generate_request
  -> Req.context_engineering
  -> PairCoordinator
  -> ScheduleBatch / ForwardBatch pair topology
```

兼容规则：

- 新客户端只写 `pair_key`，旧的 `context_engineering_pair_key` 和 `compact_pair_key` 只做 fallback，并发一次 deprecation warning。
- `pair_key` 缺失的 compact 请求在 strict mode 直接以 400/abort 拒绝，不按普通 compact 运行。
- `pair_arrival_sequence` 缺失时可在单 worker prototype 中用第一次看到 pair 的单调序号补齐；多 worker 模式必须由 runtime/router 生成并携带。
- `allow_compact_drain`、`attempt_index` 等保持 bool/int 类型，不经 `Dict[str, str]` 字符串化。
- native `/generate` 是 v1 的唯一受支持入口。OpenAI endpoint 只有在 conversion 显式传递上述结构后才标记 supported。
- native batched `/generate` 当前对整个 batch 共享一份 `custom_labels` 和一个 `routed_dp_rank`，不能表达不同 pair；CE workload 应发送独立请求，或后续把元数据改成逐 item list。

## 6. PairCoordinator

在 `python/sglang/srt/managers/pair_scheduler.py` 中保留无状态 helper，并新增 `PairCoordinator`；也可以拆为 `context_engineering/pair_coordinator.py`。推荐后者，避免继续把状态机和统计混在一个文件。

### 6.1 PairState

```python
class PairPhase(Enum):
    WAITING = auto()
    MAIN_ONLY = auto()
    PAIRED_RUNNABLE = auto()
    TOOL_SUSPEND_PENDING = auto()
    TOOL_SUSPENDED = auto()
    RESUME_PENDING = auto()
    COMPACT_DRAIN = auto()
    CLOSED = auto()

@dataclass
class PairState:
    pair_key: str
    pair_arrival_sequence: int
    main_rid: Optional[str]
    compact_rid: Optional[str]
    phase: PairPhase
    shared_prefix_group_id: Optional[str]
    allow_compact_drain: bool
    dp_rank: Optional[int]
    main_kv_location: Literal["gpu", "host", "none"]
    compact_kv_location: Literal["gpu", "host", "none"]
    pending_action: Optional[str]
```

Coordinator 至少维护：

```text
pairs_by_key: pair_key -> PairState
pair_key_by_rid: rid -> pair_key
runnable_pair_heap: (pair_arrival_sequence, pair_key)
suspended_pair_keys
closed_pair_tombstones
```

### 6.2 不变量

- 同一 pair 同时最多一个 active main physical request 和一个 active compact physical request。
- context-switch resume 可以更换 physical `rid`，但必须保留 `logical_compaction_id`、`pair_key` 和 arrival sequence。
- `PAIRED_RUNNABLE` 的两侧要么同一 iteration 都被选中，要么都不被选中。
- 非 `COMPACT_DRAIN` 状态不允许 compact-only decode。
- `TOOL_SUSPENDED` 不允许任何一侧进入 decode batch。
- pair control、cancel、OOM 和 rebootstrap 都是幂等操作；用 `switch_epoch + attempt_index` 拒绝迟到事件。
- normal scheduler 关闭 CE 时完全不创建 PairState，也不改变原调度顺序。

### 6.3 Result processing 边界

`batch_result_processor.py` 负责更新 request token/finish state。这里可以调用：

```python
pair_coordinator.observe_result(req, finish_reason, forward_iter)
```

该调用只更新状态并创建 `pending_action`，不直接 filter/release 历史 batch。下一次 `get_next_batch_to_run()` 或 `update_running_batch()` 在 live batch 上应用动作。

对于 `suspend_pair_on_finish=true` 的 main，请在 `_handle_finish_state_updated_req()` 释放 finished KV 之前转交 pair KV manager。否则现有代码会先 `release_kv_cache()`，scheduler 后续已经没有可 swap 的 main KV。

## 7. Prefill 实现

### 7.1 接入点

修改 `scheduler.py::_get_new_batch_prefill_raw()`：

1. 保持 `self.policy.calc_priority(self.waiting_queue, running_batch)` 的既有排序。
2. CE 开启时，把结果稳定分为 `Q_main` 和 `Q_compact`；高优先级类包含普通 foreground 和 CE main。
3. 队内 CE 请求使用 `pair_arrival_sequence` 保持 pair-FCFS；普通请求保留原 policy 顺序。
4. 按 `Q_main + Q_compact` 扫描并调用现有 `PrefillAdder`。
5. 不用 `compact_attention_cost_ratio` 篡改 `PrefillAdder` 的真实 token/KV 计算。

伪代码：

```python
self.policy.calc_priority(self.waiting_queue, running_batch)
if self.ce_mode.enabled:
    main_q, compact_q = partition_prefill_queue(self.waiting_queue)
    self.waiting_queue = stable_pair_fcfs(main_q) + stable_pair_fcfs(compact_q)
```

仅做 stable partition 可以作为第一步，但严格模式必须明确记录 pair arrival，不能依赖 physical retry 的 queue entry time。

### 7.2 Prefill budget

`PrefillAdder` 已经根据 `max_prefill_tokens`、chunk size、KV allocator、running decode reservation、Mamba/SWA 等实际条件做 admission。CE prefill 应优先复用这套计算：main 排在前面后，compact 自然只使用剩余容量。

若实验确实需要单独的 compact prefill quota，应新增含义明确的 `context_engineering_compact_prefill_token_budget`，按本轮 `extend_range.length` 计数，而不是整个 request `seqlen`。该 quota 只能进一步限制 compact，不能替代 `PrefillAdder` 的真实资源检查。

### 7.3 Chunk 边界抢占

在 `adder.add_chunked_req(self.chunked_req)` 前检查：

```python
if (
    self.chunked_req is not None
    and self.chunked_req.is_context_engineering_compact()
    and has_waiting_main(self.waiting_queue)
):
    park_compact_chunk_at_boundary()
```

`park_compact_chunk_at_boundary()` 必须：

- 只在上一 chunk result 已完成、`stash_chunked_request()` 已执行的安全边界调用。
- 释放或转移 radix lock_ref，避免重复 lock 和 KV leak。
- 清空 `self.chunked_req` 后重新入 `Q_compact`，保留原 `pair_arrival_sequence` 和 wait timestamp。
- 允许当前 main 成为新的 `new_chunked_req`，不触发 `assert self.chunked_req is None`。
- 不回滚已完成 chunk 的 KV；恢复 compact 时从已缓存 prefix 继续。

先为这一行为写 Scheduler-level 单测，再改代码；不能只在 AddReqResult 失败分支加 `continue`。

## 8. Decode 实现

### 8.1 两类入口都必须覆盖

统一 engine 路径中，prefill batch 在 `get_next_batch_to_run()` 合并进 `running_batch`；PD decode 路径中，新请求先经 `get_new_prebuilt_batch()` 合并，再调用 `update_running_batch()`。

因此 pair admission 的最终共同边界应放在 `update_running_batch()`，并在 PD 的 `get_new_prebuilt_batch()` 增加“未成对请求先进入 pair waiting”的早期过滤，避免 compact 先到时无意义地反复 merge/retract。

### 8.2 严格 pair 选择

用 pair group 替换当前 request keep selector：

```python
def select_decode_groups(
    running_reqs,
    pair_states,
    max_batch_pairs,
    max_physical_seqs,
    attention_budget,
) -> DecodeSelection:
    ...
```

算法：

1. 先把 request 映射为 scheduling groups；普通 foreground 是 singleton group，CE 请求按 `pair_key` 成组。
2. runnable CE groups 按 `pair_arrival_sequence` 排序。
3. 从队首选择最多 `max_batch_pairs`。
4. `PAIRED_RUNNABLE` 必须同时加入 main/compact；容量不足则整个 pair 留在 waiting/suspended，不允许只留 main。
5. `MAIN_ONLY` 只加入 main；`COMPACT_DRAIN` 只加入 compact。
6. 最后用 `ScheduleBatch.check_decode_mem(selected_indices=...)` 做实际 KV 安全检查。
7. 未选 group 如果已经占有 GPU KV，按 group retract 到可恢复队列。

模式必须显式区分：

```text
strict_pair
main_first_degraded_pair
```

`strict_pair` 用于论文主结果；`main_first_degraded_pair` 保留当前“main 不被 compact budget 阻塞”的思路，只作为消融，并记录 `pair_degraded_reason`。

### 8.3 调用顺序

推荐 `update_running_batch()` 顺序：

```text
filter finished requests
apply pending pair control/lifecycle actions
build live scheduling groups
apply strict/degraded pair selection
retract and requeue unselected groups
run existing decode memory check
if OOM: retract whole groups according to existing policy
prepare_for_decode
```

CE selection 必须发生在 `prepare_for_decode()` 前。每次 release 后统一由 Scheduler requeue，`ScheduleBatch` 不直接决定目标 queue。

### 8.4 Pair-aware OOM

保留 `_get_decode_retraction_order()`，新增：

```python
def _get_decode_retraction_groups(reqs, server_args, pair_coordinator):
    request_order = ScheduleBatch._get_decode_retraction_order(reqs, server_args)
    # 将同一 runnable pair 折叠成一个 group，并保持原 policy 的相对偏好
    return groups
```

OOM 循环按 group 处理并返回 `List[Req]`。需要覆盖：

- 只有一个二请求 pair 时不会空列表 pop。
- 一个 pair 加一个 foreground 时的 keep/retract 顺序。
- priority/length policy 的原行为对 singleton 不变。
- 两侧都出现在 retracted queue。
- host 空间不足时整组失败或 fallback，不允许只恢复一侧。

### 8.5 Compact finish 和 main finish

- compact 自然完成：PairState 转为 `MAIN_ONLY`，main 不应被 retract。
- main 因最终答案结束：runtime 发送 `CLOSE`，未完成 compact 按实验策略 cancel 或允许 drain。
- main 因 tool call 结束：进入 `TOOL_SUSPEND_PENDING`，等待 pair-specific suspend。
- 不能仅根据 `req.finished()` 推断 tool call，因为它也可能表示 EOS、length、abort 或错误。

## 9. Pair suspend/resume 与 KV

### 9.1 新控制消息

新增到 `io_struct.py`：

```python
class ContextEngineeringPairControlReqInput(BaseReq, kw_only=True):
    pair_key: str
    action: Literal["suspend", "resume", "compact_drain", "close"]
    reason: str
    switch_epoch: Optional[int] = None
    next_main_rid: Optional[str] = None
```

需要接通：

```text
HTTP endpoint
  -> TokenizerManager
  -> DataParallelController pair-affinity routing
  -> Scheduler.process_input_requests dispatcher
  -> PairCoordinator.enqueue_control
```

不要复用 `/pause_generation`；它设置 `_engine_paused` 并影响整个 scheduler。

### 9.2 Runtime 协议

agent runtime 使用固定顺序：

```text
main response parsed as tool call
-> pair_control(suspend, reason=tool_call)
-> execute/sleep for tool
-> submit next main request with same pair_key and new shared_prefix_group_id if switched
-> pair_control(resume, next_main_rid=...)
```

更稳妥的 resume 方式是把 `resume_pair=true` 和 pair epoch 放进 next main request，让 scheduler 在 main KV transfer/prebuilt 已 ready 后再原子恢复 compact，避免独立 HTTP control 与 main request 到达乱序。

### 9.3 先支持 PD decode

当前 PD decode 中存在两条不同的 host 路径：

- `Req.offload_kv_cache()` / `Req.load_kv_cache()`。
- `release_req(..., offload_kv=True)`。
- `DecodePreallocQueue.retracted_queue`。
- `resume_retracted_reqs()`。
- `DecodeKVCacheOffloadManager` 的增量 KV/Mamba archive。

前四项是 OOM retraction 的同步 CPU snapshot/restore；最后一项是完成请求或 decode 增量页写入 L3 storage 的 archive manager。后者没有 request resume API，不能直接作为 pair suspend/resume 的实现。Qwen3.5 还要求 snapshot 同时覆盖 full-attention KV 和当前 GatedDeltaNet/Mamba recurrent state；详细限制见第 17 节。

第一版建议明确限制：

```text
strict_pair + pair_kv_offload
requires disaggregation_mode == decode
requires pp_size == 1
requires overlap schedule disabled
requires page_size == 1
requires speculative_algorithm is None
requires enable_linear_replayssm == false
requires enable_unified_memory == false
```

然后新增 `resume_retracted_pair(pair_rids)`：先汇总两侧的 req slot、full KV、SWA KV 和 Mamba slot 需求，确认全部可分配后再统一 prealloc/load。任何一侧失败都不得把另一侧加入 waiting queue。

统一 engine 的 pair-specific host swap 需要新的通用 manager，不能依赖 `release_req()` 中 `disaggregation_mode == "decode"` 的条件。它应作为后续阶段实现，而不是静默退化为 recompute。

### 9.4 main turn 是新 physical request 的问题

当前 agent 每个 turn 通常发起新的 HTTP request。main 的旧 physical request 完成后，普通 result processor 会释放或写入 radix cache；这不等于论文描述的“main request 挂起并从 host 恢复”。

要严格实现并声明 main KV swap，需要：

1. 在 finished KV 被释放前创建 `PairKVSnapshot(pair_key, shared_prefix_group_id, token_ids, host_indices)`。
2. next main request 携带 `resume_from_shared_prefix_group_id`。
3. decode worker 在接收新 main 的 PD KV 前检查本地 host snapshot，只加载可复用 prefix，并仅接收/计算 tool-result 后新增 suffix。
4. token ids、model revision、cache salt、position encoding 和 generation 必须匹配；不匹配时 fallback 到普通 prefill/transfer并记录原因。

在这条链路完成前，实验只能标记为 `pair_schedule_without_main_host_resume`，不能声称实现了论文的同步 KV swap。

## 10. Serving cascade attention

### 10.1 接入范围

第一版只支持：

- FlashInfer attention backend。
- non-speculative decode。
- 非 MLA、非 SWA、非 cross-attention。
- eager execution；CUDA graph 先禁用。
- 每个 CE group 最多两个请求，同时支持普通 singleton groups。

不支持组合应 fail-fast 或明确 fallback 并计数，不能仅记录 warning 后假装启用。

### 10.2 Batch topology

Scheduler 在 `ScheduleBatch` 中生成：

```text
decode_group_ids                 # 每个 physical request 对应的 group
decode_group_q_indptr            # group -> query range
candidate_shared_prefix_lens     # 来自 pair metadata 的候选值，仅作提示
```

`ForwardBatch` 继续传递这些字段。attention backend 最终必须根据真实 token/KV layout 验证 shared prefix：

- 两个请求的 token ids 在 prefix 范围完全相同。
- 所用 KV pages 对应相同模型、cache salt 和 position。
- shared length 向 page boundary 下取整；不足一页时走 singleton/default 路径。
- 不能只因为 `pair_key` 相同就复用 KV。

### 10.3 FlashInfer plan

参考 `benchmark/cascade_kernels/bench_prefix_group_attention_kernels.py::setup_flashinfer_cascade()`，在 `flashinfer_backend.py` 新增 regular-decode cascade metadata/updater：

```text
level 0: 可选的全 batch global prefix
level 1: 每个 pair/group 的 shared prefix pages
level 2: 每个 request 的 unique suffix pages
```

没有 global prefix 时使用两层。需要构造：

```python
qo_indptr_arr
paged_kv_indptr_arr
paged_kv_indices_arr
paged_kv_last_page_len_arr
```

`forward_decode()` 在本 batch 至少存在一个有效 shared group 且 topology supported 时调用 `MultiLevelCascadeAttentionWrapper.run()`；否则保持现有 `BatchDecodeWithPagedKVCacheWrapper.forward()`。

### 10.4 正确性与性能门槛

- 相同输入下 cascade 与普通 decode logits 误差满足当前 dtype tolerance。
- greedy token ids 完全一致。
- mixed singleton/pair、不同 suffix length、partial page、pair 一侧 finish 后均有测试。
- 记录 `actual_shared_prefix_tokens`、`shared_groups`、`cascade_fallback_reason`。
- 用 Nsight/torch profiler 证明共享 prefix 没有被每个 request 重复读取；只比较 batch wall time 不足以证明机制生效。
- 复用 `benchmark/cascade_kernels` 的 case 生成和对照口径，先做 kernel parity，再做 end-to-end serving。

## 11. Inter-worker pair-density balance

### 11.1 路由所有权

同一部署只能有一个组件拥有 pair placement：

- 单个 SGLang endpoint、内部 `dp_size > 1`：在 `DataParallelController` 实现。
- 多个独立 HTTP worker 或由 sgl-model-gateway 管理的 PD 集群：在 gateway 实现。
- trace replay 可在 runtime 先选择 rank，并通过 `routed_dp_rank` 固定；这是实验简化，不是完整在线 router。

不要同时让 runtime、gateway 和 DP controller 各自重新选择 rank。

### 11.2 扩展 LoadSnapshot

在 `load_snapshot.py` 新增独立 section：

```python
class ContextEngineeringMetrics(msgspec.Struct, array_like=True):
    running_pairs: int = 0
    paired_runnable_pairs: int = 0
    running_main: int = 0
    running_compact: int = 0
    pair_waiting: int = 0
    pair_suspended: int = 0
    paired_compact_decode_tokens_window: int = 0
    all_decode_tokens_window: int = 0
    density_ewma: float = 0.0
```

`SchedulerLoadInquirer.get_loads()` 从 PairCoordinator 和 windowed counters 填充。density 必须基于时间窗口或 EWMA，不能使用进程启动以来的不衰减累计值：

```text
density_w = paired_compact_decode_tokens_window /
            max(all_decode_tokens_window, 1)
```

### 11.3 DataParallelController policy

新增：

```text
LoadBalanceMethod.PAIR_DENSITY
pair_routes: pair_key -> (dp_rank, last_seen, epoch)
```

dispatch 顺序：

1. 若请求有显式 `routed_dp_rank`，校验它与已有 pair route 一致；首次出现时可用它初始化 affinity。
2. 已知 pair 直接发到绑定 rank，绕过普通负载均衡。
3. 新 pair 从 active workers 中先按 KV usage、queue delay、total tokens 过滤过载节点。
4. 用请求携带的预计 main/compact decode work 计算 projected density。
5. 选择基础负载可接受且 projected density 最接近集群 target 的 worker。
6. pair `close` 后删除映射；同时提供 TTL 清理异常退出的 run。

一个可操作的初版 score：

```text
projected_density_w =
    (compact_work_w + expected_compact) /
    max(total_decode_work_w + expected_main + expected_compact, 1)

score_w = normalized_total_tokens_w
        + alpha * normalized_queue_delay_w
        + beta * abs(projected_density_w - target_density)
```

main 往往先于 compact 到达，因此 main 请求必须已经携带预计 compact work；否则 compact 到达时 affinity 已固定，density placement 无法纠正。trace replay 可直接使用 event 的 `decode_tokens`，在线场景使用 workload-class EWMA，并记录 prediction error。

### 11.4 Control message 路由

Pair control 不能走 `DataParallelController.send_control_message()` 的默认广播。controller 应根据 `pair_routes` 只发给 owner rank；未知或过期 pair 返回明确错误。worker 故障时不能把还持有 KV 的 active pair 任意迁移到另一 rank，必须走 abort/rebootstrap 或从持久 host snapshot 恢复。

### 11.5 Gateway 模式

若实验通过 `sgl-model-gateway`，新增 policy module（例如 `sgl-model-gateway/src/policies/pair_density.rs`），在 `policies/factory.rs`、`config/types.rs` 和 CLI choices 注册。HTTP/PD router 必须从 native `/generate` body 的 CE 元数据提取 pair key，并将选中的 DPAware rank 注入请求。

当前 gateway 仍注入兼容字段 `data_parallel_rank`；SGLang 会迁移为 `routed_dp_rank`。新代码应优先改为 canonical 名字，并保留旧名兼容测试。

## 12. 可观测性

现有 `context_engineering_batch` 继续保留，但需要增加事件和计数。

### 12.1 调度事件

```text
context_engineering_pair_registered
context_engineering_pair_admitted
context_engineering_pair_degraded
context_engineering_pair_suspended
context_engineering_pair_resumed
context_engineering_pair_retracted
context_engineering_pair_closed
context_engineering_cascade_batch
context_engineering_pair_placement
```

公共字段：

```text
pair_key, pair_arrival_sequence, shared_prefix_group_id
logical_compaction_id, attempt_index, switch_epoch
main_rid, compact_rid, old_phase, new_phase, reason
dp_rank, forward_iter, kv_location, actual_shared_prefix_tokens
```

### 12.2 核心指标

- `pair_coscheduled_iteration_ratio`：分母只统计 main/compact 同时 runnable 的 iterations。
- `pair_fcfs_inversion_count`。
- `pair_suspend_sync_ratio`、`pair_resume_sync_ratio`。
- `orphan_compact_decode_iterations`：除 `COMPACT_DRAIN` 外必须为 0。
- `compact_drain_iterations`。
- main/compact prefill queue delay。
- `compact_prefill_preempted_by_main_count`。
- pair KV offload/load bytes 和 latency。
- `actual_shared_prefix_tokens` 与 candidate 值偏差。
- cascade hit/fallback count 及 fallback reason。
- worker `density_ewma` 的 max-min、标准差和 CV。

日志中不能用未哈希的高基数 `pair_key` 作为 Prometheus label；它只进入结构化日志/trace。Prometheus 使用聚合计数和低基数 reason/mode。

## 13. 配置建议

建议把当前 bool 开关演进为明确模式，同时保留兼容 alias：

```text
--pair-scheduler-mode disabled|main_first_degraded_pair|strict_pair
--context-engineering-max-batch-pairs N
--context-engineering-max-physical-seqs N
--context-engineering-decode-attention-token-budget N
--context-engineering-compact-attention-cost-ratio R
--context-engineering-enable-pair-kv-offload
--context-engineering-enable-cascade-attention
--context-engineering-pair-density-weight BETA
--context-engineering-pair-route-ttl-seconds N
```

规则：

- `--enable-pair-scheduler` 暂时映射到 `main_first_degraded_pair` 并发 deprecation warning，避免旧实验静默变成 strict 语义。
- 论文主实验必须显式传 `strict_pair`。
- `compact_attention_cost_ratio` 只改变 soft admission cost，不改变真实 KV 分配和 kernel work。
- 所有整数非负，ratio 有明确范围；非法组合启动时失败。
- strict 初版仅支持的 backend/PP/overlap 组合必须在 `ServerArgs` 校验阶段 fail-fast。

## 14. 实施顺序

### Commit 1：恢复可运行基线并加 Qwen3.5 capability gate

文件：

- `scheduler.py`
- `schedule_batch.py`
- `kv_cache_builder.py`
- `scheduler_pp_mixin.py`

工作：

- 恢复 `_record_step_counters`、`_get_decode_retraction_order`、`build_rebootstrap_payload` 和 Ascend import。
- rebootstrap payload 保留 CE typed metadata/custom labels。
- 删除/禁用所有未实现方法调用。
- 恢复 `running_batch` 参数使用。
- 用 capability gate 替代 hybrid SWA/SSM 的无条件 warning；只放行第 17 节已验证的 Qwen3.5 组合。
- 撤掉 compact AddReqResult 特殊 `continue`，直到正式 prefill policy 接入。

验收：所有原有 CPU tests 通过，CE flag 关闭时普通 server smoke 能生成 token。

### Commit 2：类型化请求契约

- 新增 `ContextEngineeringRequestInfo`。
- native `/generate` 校验和兼容字段归一化。
- 全链路透传到 Req、session 和 PD rebootstrap。
- 修正 canonical pair key 和 generation 分离。
- 增加 malformed/duplicate/old alias/batched input tests。

### Commit 3：prefill 双队列

- 在 `_get_new_batch_prefill_raw()` 接入 stable two-class queue。
- 保持原 policy 和 `PrefillAdder` 资源语义。
- 实现 compact chunk boundary parking/preemption。
- 加入 queue delay 和 preemption metrics。

### Commit 4：single-worker decode PairCoordinator

- 新增 PairState/Coordinator。
- unified 和 PD decode 都从 `update_running_batch()` 使用 group selection。
- strict 与 degraded 模式分开。
- 重写 OOM 为 group retract，并保证 requeue。
- 实现 `COMPACT_DRAIN` 明示例外。
- 初版限制 `pp_size=1`、disable overlap；跑通后再扩 PP/overlap。

### Commit 5：pair control 和 PD host KV lifecycle

- 增加 pair control API/IPC/dispatcher。
- 实现 pair-specific suspend、atomic resume 和 close。
- 在 finished KV release 前保存需要的 snapshot。
- Qwen3.5 snapshot 必须同时保存 full-attention KV 和当前 GDN/Mamba state，并按 TP/PP rank 汇总 ack。
- 接通 agent runtime 的 tool-call suspend/resume 和 hard-limit drain action。
- 测试迟到 control、重复 control、cancel/finish race、host OOM。

### Commit 6：serving cascade attention

- ScheduleBatch/ForwardBatch 传 group topology。
- FlashInfer updater 构造 group/unique page tables。
- eager regular decode 接入 `MultiLevelCascadeAttentionWrapper`。
- 正确性、fallback、mixed topology 和 profiler 验证。
- 完成前所有结果标记 `scheduler_only`。

### Commit 7：pair affinity 和 density routing

- 扩 `LoadSnapshot` 和 `SchedulerLoadInquirer`。
- DataParallelController 或 gateway 只选一个 placement owner。
- 实现 affinity map、density score、TTL 和 pair control direct routing。
- 多 DP worker 做 skewed density replay。

### Commit 8：PP/overlap 和完整回归

- PP ranks 对 pair selection、control epoch 和 retraction group 达成一致。
- overlap 模式只在 pending action 已应用到 live batch 后继续 launch。
- speculative/SWA/MLA 等仍不支持的组合保持 fail-fast 或可观测 fallback。

## 15. 测试计划

### 15.1 当前已执行的审计测试

host Python 没有安装 pytest。`compileall` 使用临时 pycache 后通过，只能证明语法正确：

```bash
PYTHONPYCACHEPREFIX=/tmp/sglang_ce_pycache \
python3 -m compileall -q \
  python/sglang/srt/managers/pair_scheduler.py \
  python/sglang/srt/managers/scheduler.py \
  python/sglang/srt/managers/schedule_batch.py
```

运行中的 `lmsysorg/sglang:v0.5.17-cu129` 容器可见 4 张 A100。第一次未设置 `PYTHONPATH` 时，pytest 错误地导入了镜像内已安装 SGLang，得到假阳性的 `41 passed`。

明确加载工作区源码后：

```bash
cd /data/xjzhang/context_engineering_serving/references/sglang_dev
PYTHONPATH=$PWD/python python3 -m pytest -q \
  test/srt/test_pair_scheduler.py \
  test/registered/unit/managers/test_retraction_order.py \
  test/registered/unit/managers/test_priority_scheduling_disaggregation.py
```

加入 `test_decode_kvcache_offload_hybrid.py` 后，实际结果为 `6 failed, 39 passed, 3 subtests passed`：

- 5 项失败来自 `_get_decode_retraction_order` 缺失。
- 1 项失败来自 `Req.build_rebootstrap_payload` 缺失。
- 新增的 4 项 hybrid offload helper 单测通过，但它们通过 `object.__new__` 绕过 manager 初始化，也没有执行 Qwen3.5、D2H/H2D、storage hit、request finish 或 retract/resume 流程，不能证明功能可运行。

后续所有容器测试必须显式设置 `PYTHONPATH=$PWD/python`，或先确认容器内 editable install 正确指向当前 worktree。

### 15.2 必需 CPU/unit tests

除恢复现有测试外，新增：

- Scheduler module import 和关键方法存在性测试，避免 pure helper direct-import 掩盖集成错误。
- normal scheduler flag-off 行为与基线一致。
- prefill main/compact stable ordering。
- compact chunk 被后来 main 在下一边界抢占。
- strict pair admission 不拆 pair。
- degraded pair明确产生 reason。
- compact-only 仅 `COMPACT_DRAIN` 放行。
- pair OOM group retract/requeue，不发生空列表 pop。
- duplicate main/compact side 被拒绝。
- suspend/resume/close 幂等与 epoch 去重。
- PD rebootstrap 保留 typed metadata。
- DP affinity、route mismatch、TTL 和 projected density。
- load snapshot msgpack round-trip。

### 15.3 GPU smoke

每个阶段至少验证：

1. flag off：普通 main 请求完成。
2. degraded mode：main 与 compact 能完成且无 request loss。
3. strict mode：两侧同时 runnable 时 decode iteration group 一致。
4. hard-limit：显式 compact drain 能完成并恢复 main。
5. OOM：pair 同步 retraction/resume。
6. PD：rebootstrap 后 CE identity 不丢失。
7. host swap：日志显示两侧所需 KV 的 device-to-host 和 host-to-device 完成。
8. cascade：输出与 baseline 一致，实际 shared prefix 非零。

GPU 测试必须在 CUDA 12.9 的 `lmsysorg/sglang:v0.5.17` 容器中运行，并显式加载 host 的 customized repo。若容器内 `nvidia-smi -L` 看不到设备，先重启容器再测。

### 15.4 实验验收门槛

只有同时满足以下条件，结果才可以标为论文 `pair_scheduler`：

- `pair_fcfs_inversion_count == 0`。
- 同时 runnable pair 的 strict coschedule ratio 达到预设阈值，且所有例外有 reason。
- `orphan_compact_decode_iterations == 0`，compact drain 单独统计。
- tool-call suspend/resume 有成对状态和 KV location 证据。
- 同 pair 固定到一个 decode worker。
- 多 worker 实验报告 density 分布；单 worker 标记 `not_applicable`。
- serving cascade 路径报告真实 `actual_shared_prefix_tokens > 0`，并通过 correctness test。
- flag-off baseline 无吞吐和正确性回归。

否则应使用更准确的标签：

```text
metadata_only
scheduler_helper_only
main_first_degraded_pair
strict_pair_scheduler_only
pair_schedule_without_main_host_resume
strict_pair_with_cascade
```

## 16. 暂不混入本实现的改动

- multimodal PD rebootstrap。
- speculative decode 下的 pair cascade topology。
- 同一 native batched request 内逐 item 路由不同 pair/rank。
- pre-compaction merge 的语义决策；scheduler 只接受 runtime 已合并后的唯一 active compact side。

Qwen3.5 hybrid SSM decode radix/HiCache 已成为本方案的前置工作，移至第 17 节。其余工作都需要独立设计和测试。尤其不要为了运行 CE workload 而删除原有兼容性检查或改变普通 scheduler 行为。

## 17. Qwen3.5 + PD decode HiCache retraction 专项方案

本节针对当前新增约束：模型为 Qwen3.5 系列，采用 PD 分离部署，decode worker 必须支持 HiCache 和 request retraction，并最终服务于 strict pair 的同步挂起/恢复。

### 17.1 先统一术语：当前代码有四种不同生命周期

以下四件事不能继续都称为 “offload/retract”：

| 场景 | 触发条件 | 旧 KV 是否可复用 | 当前入口 | 目标动作 |
| --- | --- | --- | --- | --- |
| decode OOM retraction | 下一 decode step 无显存 | 可复用，权重不变 | `ScheduleBatch.retract_decode()` | 保存当前精确状态、释放 L1、稍后本地恢复 |
| CE pair suspend | main 进入 tool call | 可复用，权重不变 | 尚无 pair control | main/compact 成对保存和成对恢复 |
| weight-update rebootstrap | `pause_generation(mode="retract")` 后更新权重 | 不可复用旧 KV | `hold_rebootstrap()` | 丢弃旧 KV，回 prefill 用新权重重算 |
| completed-request archive | 请求结束或增量达到 stride | 作为未来 prefix cache | `DecodeKVCacheOffloadManager` | 按内容 hash 写 L2/L3，不恢复原 Req |

`346934b2f1ab5` 主要扩展的是最后一行：让 completed/incremental archive 可以附带一个 Mamba checkpoint。它不是 OOM retraction 或 pair suspend 的完整实现。实施时必须给四条路径不同的 API、状态和指标，不能仅用 `--disaggregation-decode-enable-offload-kvcache` 判断。

### 17.2 `346934b2f1ab5` 已完成的部分

该提交提供了有价值的底层积木：

- `DecodeKVCacheOffloadManager` 能识别 `HybridLinearKVPool`。
- 为 full-attention KV 和 Mamba state 构造 `HostPoolGroup([KV, MAMBA])`。
- hybrid 路径使用 `HybridCacheController`，Mamba 以 `PoolTransfer(PoolName.MAMBA, TRAILING_PAGES)` 随最后一个 page hash 归档。
- fixed `hicache_size` 会在 full KV 与 Mamba host pool 间按 device pool 字节数拆分。
- request finish 后增加 `free_mamba_cache(req)`，storage ack 后增加 side-pool host slot 释放。
- `UnifiedRadixCache` 补上 decode mixin 所需的 `is_load_back_event_done()` 方法。

这些改动说明现有 `HybridCacheController`/`HostPoolGroup` 可以复用，不应再实现第三套 Mamba D2H/H2D kernel。但是当前只验证了 helper 的局部行为。

### 17.3 阻止 Qwen3.5 正常运行的问题

#### QP0-1：finished request 释放路径必然访问不存在的字段

`decode_kvcache_offload_manager.py::_release_finished_req()` 直接执行：

```python
self.tree_cache.protected_size_ -= len(req.prefix_indices)
```

Qwen3.5 hybrid 使用 `UnifiedRadixCache`；其 protected size 存在 `tree_core.component_protected_size_` 中，没有 `protected_size_` 字段。因此只要启用 `--disaggregation-decode-enable-offload-kvcache` 并完成一次请求，该路径就可能以 `AttributeError` 结束。

修复不能是在 `UnifiedRadixCache` 上伪造同名整数。manager 不应手工修改 radix lock accounting；应调用 cache 的公共 `dec_lock_ref()`/release API，且只释放 Req 实际持有的 node handles。更稳妥的方案是把 finished archive 的 release 提交给 `UnifiedRadixCache`，由 tree component 完成 FULL/MAMBA ownership 转移。

#### QP0-2：L3 storage hit 查询仍是明确的 `NotImplementedError`

`UnifiedRadixCache.query_storage_hit_length()` 在 `enable_storage=True` 时直接抛异常。PD decode admission 的 `_build_decode_prefix_match()` 在 host node 已 backup 时会调用它。因此 Qwen3.5 + decode radix + hierarchical cache + storage 尚未形成 read path 闭环。

在该方法实现前，不允许把 Qwen3.5 L3 配置标为 supported。短期只放行 L1/L2，令 `hicache_storage_backend=None`；长期应移植 `HiRadixCache.query_storage_hit_length()` 的流程，并用 `HybridCacheController.batch_exists_v2` 同时求交：

```text
usable_hit_pages = min(KV contiguous hit pages,
                       compatible trailing Mamba checkpoint coverage)
```

返回的 hit 长度必须保证存在一个与该 KV prefix 末端完全对齐的 Mamba state；只有 KV 命中而 Mamba miss 时不能跳过 prefill。

#### QP0-3：两个 controller 重复拥有同一 device pool

启用 decode radix HiCache 时，`UnifiedRadixCache.init_hicache()` 已创建并注册一个 `HybridCacheController`。再启用 decode archive，scheduler 又创建第二个 `DecodeKVCacheOffloadManager` 和第二个 controller：

- 两者各自分配一整套 KV/Mamba host pool，实际 host 内存可能接近配置值的两倍。
- 两者都对同一 `HybridLinearKVPool` 和 `HybridReqToTokenPool` 调用 `register_layer_transfer_counter()`，后创建的 manager 会覆盖前一个 counter。
- tree load-back 与 archive 的 ack queue、layer readiness 和 slot ownership 没有统一仲裁。

目标结构应是一个 worker 只有一个 `HybridCacheController`。`UnifiedRadixCache` 是 owner；archive 和 retract manager 通过类型化 operation 调用它，而不是各自创建 controller/host pool。合并前应拒绝同时开启 `disaggregation_decode_enable_radix_cache`、`enable_hierarchical_cache` 和 `disaggregation_decode_enable_offload_kvcache` 的 Qwen3.5 配置。

#### QP0-4：当前 CE/OOM retraction 主入口自身仍不可运行

当前 `ScheduleBatch.retract_decode()` 调用已缺失的 `_get_decode_retraction_order()`；PD rebootstrap 调用已缺失的 `Req.build_rebootstrap_payload()`。这与新增 hybrid helper 是否通过无关，必须先按 Commit 1 恢复。

此外当前 pair OOM 逻辑在仅剩一个两成员 pair 时可能把 index list 清空，随后 last-request 分支继续 `pop()`；这会把 OOM 变成 `IndexError`。必须按 scheduling group 而不是 request index 重写。

### 17.4 尚未闭环的正确性问题

#### QP1-1：archive manager 没有恢复原 request 的 API

`DecodeKVCacheOffloadManager` 只有 write/backup/finalize，没有按 `rid` 或 snapshot id 分配新 device slots、load Mamba、恢复 `req_to_token` 并重新入队的接口。`resume_retracted_reqs()` 仍只读取 `Req.kv_cache_cpu`。因此 archive manager 不能被 PairCoordinator 当作 suspend/resume store。

#### QP1-2：CPU retraction 只在受限配置下可能正确

当前 OOM 路径的 `Req.offload_kv_cache()` 会调用 `HybridLinearKVPool.get_cpu_copy()`，能够复制 full KV 和 active Mamba `conv/temporal` state；这是第一阶段可复用的最短路径。但仍有以下限制：

- `UnifiedMambaTokenToKVPoolAllocator` 没有实现 `get_cpu_copy/load_cpu_copy`，会落到 base `NotImplementedError`。
- speculative/ReplaySSM 的 ring/intermediate buffer 不属于 transferable state；仅复制 cursor 或 checkpoint 不能重建未 fold 的 ring。
- ping-pong checkpoint slots 和对应序列长度没有进入 snapshot；恢复后必须显式重建 tracking metadata。
- `HybridReqToTokenPool.alloc()` 将新 active slot 标记为 `mamba_needs_clear=True`；H2D 恢复成功后必须置回 `False`，避免后续 extend/mixed path 清掉刚恢复的状态。
- `reset_for_retract()` 没有完整定义 lazy checkpoint 标志、PD rebootstrap 字段和 CE pair epoch 的重置/保留规则。

所以第一阶段只允许非 speculative、非 ReplaySSM、非 unified-memory 的 Qwen3.5，并对其他组合启动时报错。

#### QP1-3：Mamba checkpoint 与 KV prefix 的原子性不足

`_build_mamba_backup_transfers()` 只在 `mamba_last_track_seqlen == chunk_end` 时归档 ping-pong checkpoint，这个对齐判断是必要的，但不充分：

- active decode 可在 D2H 尚未完成时继续运行，后续 track boundary 可能复用同一 ping-pong slot。
- manager 没有对 Mamba device slot 加 pin/refcount，也没有复制到专用 immutable staging slot。
- `OffloadedState.inc_len` 在 D2H ack 前即推进；失败、取消和重试没有事务状态。
- storage thread 异常没有 request-level failure ack 和 timeout，可能留下永久 inflight/host slot。

应为 Mamba checkpoint 增加 immutable ownership：要么先 COW 到专用 staging slot再异步写，要么 pin 住源 slot直到 D2H event 完成。operation 状态至少包含 `PREPARING -> D2H_INFLIGHT -> HOST_COMMITTED -> RELEASED/FAILED`，只有 `HOST_COMMITTED` 后才能释放对应 L1 state。

#### QP1-4：storage key namespace 未与正常 radix 路径证明一致

manager 直接从 `origin_input_ids + output_ids` 计算 hash chain，只记录 `rid -> OffloadedState`，没有显式携带 `req.extra_key`、cache salt、LoRA、model/weight epoch、pair generation 或 multimodal identity。即使底层 backend 另有 model namespace，这条自建 hash 链也必须通过测试证明与 prefill/UnifiedRadixCache 的 key 完全一致。

建议删除 manager 内独立 `_compute_prefix_hash()` 语义，让 `UnifiedRadixCache` 从同一个 `RadixKey`/node API 产生 keys。snapshot 恢复则不用 content hash 查找，使用不可猜测的 `snapshot_id + model_epoch + rank`。

#### QP1-5：PP/TP/overlap 没有完成全局提交

新 manager 只在传入的 TP group 上对 ack 数取 MIN，没有 PP group；pair retraction 也没有等所有 PP stages 的 FULL/MAMBA 保存完成。任一 stage 提前释放就会造成下一次 forward 状态不完整。

overlap 下还有一个已 launch、尚未 result-process 的 batch。snapshot 必须在 scheduler quiescent point 创建：先停止新 launch，处理/取消 last batch，记录 forward-done event，再读取 `kv_committed_len` 和 Mamba state。不能仅以 Python `Req.output_ids` 长度推断已提交 KV。

### 17.5 第一阶段支持矩阵

先定义一个可以端到端验证的窄基线，随后逐项扩展：

| 维度 | 第一阶段 | 未满足时行为 |
| --- | --- | --- |
| 模型 | Qwen3.5 dense/MoE text backbone，hybrid full attention + GDN | 其他 hybrid 架构单独验收 |
| 部署 | PD，decode side | 非 decode 不启用本地 retract store |
| parallel | `pp_size=1`，同构 TP | fail-fast |
| scheduler | overlap disabled | fail-fast |
| page | `page_size=1` | fail-fast，避免 tail page snapshot 语义 |
| speculative | disabled | fail-fast |
| ReplaySSM | disabled | fail-fast |
| unified memory | disabled | fail-fast |
| HiCache | L1/L2；单一 `UnifiedRadixCache` controller | 不创建第二个 controller |
| L3 storage | disabled | 实现 hybrid hit-query 后再放行 |
| multimodal input | 暂不作为 rebootstrap 基线 | fail-fast 或走普通新请求 |

Qwen3.5 text-only checkpoint 的 architecture 名称还需纳入同一 override/capability registry；当前 `_qwen3_5_hybrid_overrides` 只列出 conditional-generation 名称和 `Qwen3NextForCausalLM`，必须补齐并新增参数解析单测，避免 dense/MoE text 入口得到不同 page/backend 默认值。

### 17.6 精确 retraction snapshot 契约

新增类型化对象，不能继续在 `Req` 上动态挂 `kv_cache_cpu`：

```python
class HybridRetractSnapshot(msgspec.Struct):
    snapshot_id: str
    rid: str
    pair_key: str | None
    pair_epoch: int | None
    model_epoch: int
    kv_committed_len: int
    token_ids_digest: str
    full_kv: object
    mamba_state: object
    mamba_state_seqlen: int
    pending_output_id: int | None
    source_tp_rank: int
    source_pp_rank: int
    state: str
```

核心不变量：

- `full_kv` 精确覆盖 `req_to_token[req_pool_idx, :kv_committed_len]`。
- `mamba_state` 是处理完相同 `kv_committed_len` 个 token 后的 active recurrent state，不是较旧的 radix checkpoint。
- `mamba_state_seqlen == kv_committed_len`；不相等时 snapshot 不可提交。
- `pending_output_id` 表示已采样但 KV 尚未写入的边界 token，不能混入 committed prefix。
- `model_epoch` 不变才允许本地恢复；权重更新必须走 rebootstrap。
- snapshot 只有收到全部 TP/PP rank 的 `HOST_COMMITTED` 后才对外可见。

第一阶段可以让 `full_kv/mamba_state` 继续使用 pinned CPU tensor 结构，以最少改动证明正确性；第二阶段再把它换成 `HostPoolGroup` indices，Req/PairCoordinator API 不变。

### 17.7 单请求 retract/restore 时序

#### Retract

```text
decode memory gate selects a retraction group
-> stop the group from entering the next batch
-> wait forward-done / drain overlap result
-> freeze kv_committed_len and pending token
-> allocate host snapshot capacity for FULL + MAMBA
-> D2H both components
-> all TP/PP ranks report HOST_COMMITTED
-> release_kv_cache(..., is_insert=False)
-> reset only transient scheduling fields
-> enqueue snapshot handle in retracted_queue
```

若 host allocation 或任何 rank 的 D2H 失败，在释放 device state 之前回滚 host allocation并保留 request；不能出现“FULL 已释放、MAMBA 未保存”的半提交。

#### Restore

```text
peek retracted group
-> reserve req slots + FULL token slots + MAMBA active slots for whole group
-> write req_to_token mapping
-> H2D FULL + MAMBA
-> wait all TP/PP completion
-> set kv_committed_len/mamba_state_seqlen
-> set mamba_needs_clear=False and rebuild tracking fields
-> atomically move whole group to waiting/running eligibility
-> release host snapshot only after first resumed batch owns device state
```

容量检查必须同时考虑 full KV token slots、每个 request 的 active Mamba slot、启用 extra-buffer 后的 tracking slots，以及 reserved decode tokens。不能只用 `token_to_kv_pool_allocator.available_size()`。

### 17.8 strict pair 的原子扩展

PairCoordinator 调用 `retract_group(pair_key)`，而不是循环调用两次 `release_req()`：

1. 解析并验证 pair 恰有一个 main 和一个 compact；允许论文明确的 `MAIN_ONLY`/`COMPACT_DRAIN` 状态。
2. 预留两侧 host snapshot 容量。
3. 并行发起两侧 FULL/MAMBA D2H。
4. 所有 rank、两个 member 均成功后，单次提交 `PairSnapshot` 并释放 L1。
5. 任一失败则整体回滚；不得只把一侧加入 `retracted_queue`。

resume 同样先做全组 capacity reservation。若 pair 暂时无法整体恢复，保留在 pair-level queue，不允许 main 单独越过更早到达且已 ready 的 pair；论文允许的 main-only 和 compact-drain 必须由 PairState 显式产生，不能由 allocation 失败隐式产生。

对于 tool-call suspend，runtime 等待 scheduler 返回 `pair_epoch + snapshot_id + HOST_COMMITTED` 后才能认为 suspend 完成。下一 main turn 携带该 epoch；过期、重复或跨 worker 的 resume 被幂等拒绝。

### 17.9 建议的代码改动

#### A. 先恢复基线

- `schedule_batch.py`：恢复 `_get_decode_retraction_order()` 和 `Req.build_rebootstrap_payload()`。
- `scheduler.py`：恢复缺失 import/counter，移除未实现的无条件调用。
- `kv_cache_builder.py`/`server_args.py`：实现上表 capability gate。

#### B. 分离 archive 与 retract

- 新增 `disaggregation/decode_retract_manager.py`，定义 snapshot/group transaction；第一版封装已有 CPU copy。
- `release_req()` 接受明确的 `release_mode=DROP|SNAPSHOT|REBOOTSTRAP`，禁止 bool `offload_kv` 同时表达三种语义。
- `DecodePreallocQueue.retracted_queue` 保存 snapshot handle，不把大 tensor 动态挂在 Req 上。
- `resume_retracted_reqs()` 改成 prepare/commit 两阶段，增加 `resume_retracted_group()`。

#### C. 修复 hybrid HiCache owner

- `UnifiedRadixCache` 保持唯一 `HybridCacheController` owner。
- 将 archive operation 接入 tree cache 公共 API，并删除 Qwen3.5 路径中第二套 host pool/controller。
- 用 tree API 完成 lock/ref accounting，删除对 `protected_size_` 的直接写。
- 为 operation 增加 kind、epoch、error ack、timeout、cancel 和幂等 cleanup。

#### D. 完成 L3 read path

- 实现 `UnifiedRadixCache.query_storage_hit_length()`。
- storage exists/prefetch/load-back 同时携带 FULL 和 MAMBA transfer。
- 验证 Mamba trailing key 对齐；KV hit 与 Mamba miss 时缩短到最近完整 checkpoint，剩余 suffix 回 prefill 重算。
- key namespace 纳入 cache salt/LoRA/model epoch，并验证与 prefill 端一致。

#### E. 接入 PairCoordinator

- OOM selector 返回 retraction groups。
- pair control 调用 group snapshot transaction。
- PP/TP 用 group operation id 达成一致后再提交。
- metrics 分开记录 OOM、tool suspend、rebootstrap 和 archive。

### 17.10 验证顺序和测试矩阵

当前新增测试只有 4 个 CPU helper case，远不足以宣称支持 Qwen3.5。按以下顺序补齐：

1. **结构单测**：真实构造 lightweight `UnifiedRadixCache + HostPoolGroup`，确认只存在一个 controller、FULL/MAMBA pool 容量和 layer mapping 正确。
2. **snapshot round-trip**：给 KV 和 GDN conv/temporal 写确定值，retract、覆写 device slot、restore 后逐 tensor bitwise compare。
3. **token continuation**：同一 prompt 固定 seed，比对不 retract 与指定 step retract/resume 后续 token/logprob。
4. **OOM group test**：只剩一个 pair、host allocation failure、第二 member D2H failure、重复 resume、abort while retracted，均不得丢请求或拆 pair。
5. **PD transfer test**：prefill -> decode 首次 Mamba transfer、decode retract/restore、再次生成；检查 `kv_committed_len` 和 Mamba sequence alignment。
6. **L1/L2 HiCache test**：新请求命中 host prefix，FULL 与 MAMBA 一起 load-back；Mamba 缺失时必须 fallback/recompute。
7. **L3 test**：完成 `query_storage_hit_length` 后才加入；覆盖 storage partial hit、missing trailing checkpoint、backend error 和 key namespace。
8. **分布式 test**：TP=2，然后 PP=2；注入单 rank failure，验证全局不提交。
9. **overlap test**：最后加入，分别在 result pending、track boundary 和 pair suspend 到达时触发。

第一阶段 GPU smoke 使用 CUDA 12.9 的 `lmsysorg/sglang:v0.5.17` 容器，并显式 `PYTHONPATH=$PWD/python`。建议固定小 Qwen3.5 checkpoint和以下对照：

```text
A: PD baseline, radix/HiCache off
B: PD + decode radix + L1/L2 HiCache
C: B + forced single-request OOM retract/restore
D: B + forced pair OOM retract/restore
E: B + tool-call pair suspend/resume
```

每组至少比较完整 token ids、首个恢复 token、logprob、KV/Mamba allocator 可用量回到基线、host slot 无泄漏。只有 C/D/E 正确且连续压力循环通过后，才能把 Qwen3.5 标记为 `retract_supported`；`346934b2f1ab5` 当前只能标记为 `hybrid_decode_archive_write_prototype`。
