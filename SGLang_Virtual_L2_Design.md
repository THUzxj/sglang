# SGLang Virtual L2：基于有界 Pinned Staging Buffer 与真实 DMA 的大容量 Host Cache 性能模拟

**技术设计文档（Implementation Design / RFC）**
**版本：** v1.0 · 2026-10-16
**源码基线：** `sgl-project/sglang` `main`，查询时 HEAD `06934575b9eb0620832e842facc0591b5a3e9b3f`（建议开发时固定 commit；本文中新增的标识均为提案，不是现有 API）。
**目标：** 在物理 Host RAM 有限的机器上，运行真实 SGLang Scheduler、GPU Forward、L1↔L2 DMA 与 PD Transfer，以性能实验形式评估远大于实际 RAM 的 L2 Cache。**一次性覆盖 MHA/GQA、MLA、Gated DeltaNet/Mamba、Hybrid Pool 与 PD Disaggregation。**
**性质：** 性能模拟，不保证 Fake L2 命中后生成内容语义正确；不用于部署、准确率评测或真实 L3 数据持久化。

## 1. 设计摘要

使用两个完全分离的空间：**逻辑 L2（Virtual Host Pool）**记录前缀/状态、容量、分配、驱逐及传输计量；**物理有界 L2（Pinned Staging Pool）**只为在途 DMA 分配较小的、可复用的 Host Tensor。写入时对真实 GPU 缓存发起 D2H 拷贝，完成后丢弃 staging 中的数据；读取时从预初始化的合成 Host 缓冲区发起同等字节数的 H2D 拷贝，将数据写入真实 GPU KV/SSM 槽位。这保留数据搬运、Kernel 启动、CUDA Stream/Event、硬件竞争与 GPU Forward，但不复原原始上下文对应的数值。

**核心边界：** Virtual L2 不是 `--disaggregation-transfer-backend fake`，也不是现有 `--hicache-host-memory-mode buffer_only`：前者是不同的 PD 转移路径，后者要求 L3 并且在 Decode 端存在限制。Virtual L2 必须作为独立的 Host Pool/Transfer Mode 接入。

**验收判定：** 逻辑容量可以配置为 256/512/1024 GB；进程持有的额外物理 pinned Host RAM 严格有界；读/写传输实际占用 GPU↔Host 带宽；无过早 ACK、重用未完成缓冲区、GPU 槽位抢占；MHA、MLA、Mamba/DeltaNet 与 PD 的预定测试用例全部通过。

## 2. 非目标与可信度边界

1. **不保证生成质量**：被淘汰的 KV/SSM 本体已被丢弃，L2 hit 恢复的是合成状态。模型仍运行真实 Forward，但 Attention、MoE 路由、采样、Speculative Acceptance 可能受到数值扰动。
2. **不保证逐请求 TTFT 与真实大 DRAM 完全一致**：可观测的是同类计算图、数据流量及模拟带宽模型下的系统性能；缺失真实 512GB 内存的 NUMA、页迁移、TLB、Host 内存通道竞争和真实大地址空间 pinning 效应。
3. **不把 Fake 数据传入生产 L3**。默认禁用 L3 写入/读取；如必须研究含 L3 的调度，应启用独立命名空间的 Shadow Storage，并仅宣称元数据/时延模拟。
4. **不自动模拟网络 PD 带宽**。Prefill→Decode 使用原有 Mooncake/NIXL 等真实路径；Virtual L2 只处理每个 worker 私有的本地 L1↔L2。若设备/网络不足以复现目标平台，应另行校准。
5. **不把静态 CPU `sleep` 视为 DMA**：必须实际提交同量级、同方向、同布局的 DMA，额外排队/带宽参数只能在不破坏 CUDA 依赖的前提下调整。

## 3. 当前 SGLang 代码映射（已核对）

| 源码 / 类 | 目前职责 | Virtual L2 改动策略 |
|---|---|---|
| `mem_cache/pool_host/base.py` / `HostKVCache` | `size/page_num/free_slots/alloc/free`，真实 Host Buffer、物理搬运接口 | 复用 Slot 语义；将虚拟容量与真正的 Host Tensor 分离 |
| `pool_host/mha.py` / `MHATokenToKVPoolHost` | MHA/GQA K/V、page/layer layout、传输 Kernel | 用小容量物理 Stage Host Pool 复用原拷贝 Kernel |
| `pool_host/mla.py` / `MLATokenToKVPoolHost` | MLA 维度、DCP、draft/indexer、dummy 模式 | 复用 MLA 逻辑索引；另建 Stage Pool，不直接把 `is_dummy` 当可传输 Fake Pool |
| `pool_host/mamba.py` / `MambaPoolHost` | temporal、conv、额外 checkpoint side state，独立 Host 分配 | 实现真正的虚拟 allocator + 每类状态的 staging；保留 side-state 次序 |
| `pool_host/group.py` / `HostPoolGroup`、`PoolEntry` | Anchor 与 side pools 的组合、分配回滚、Layer Mapper | 每个 Pool Entry 独立预算/几何；组合操作整体 ACK |
| `mem_cache/l2_transfer.py` / `L2TransferEngine` | `submit_device_to_host`、逐层 `submit_host_to_device`、TransferCompletion | 增加 virtual 分支、分块、缓冲租约、提交/完成 Event |
| `managers/cache_controller.py` / `HiCacheController` | `CacheOperation`、write/load 队列、ACK、LayerLoadingEvent | 尽量保留 API，扩展待完成分块关联与取消清理 |
| `hybrid_cache/hybrid_cache_controller.py` / `HybridCacheController` | 多 Pool 协同与储存路径 | 保留 `PoolTransfer` 及其 PoolName、Layer Mapper |
| `hybrid_cache/hybrid_pool_assembler.py` | Host Pool 创建与 HostPoolGroup 装配 | 统一工厂开关，替换为 Virtual Host Pool + Stage Backend |
| `unified_radix_cache.py` / `UnifiedRadixCache` | Prefix 索引、L1/L2 驻留、evict/demote/lock | 原则上不修改缓存决策；仅 Host 侧物理实现替换 |
| `disaggregation/decode_kvcache_offload_manager.py` | Decode 新增 token KV 异步写 Host，再可能转 L3 | 接入 virtual write、ACK 与 retraction 防竞态 |
| `disaggregation/decode_hicache_mixin.py` | Decode prefix match、local restore、load-back gating | 保留先恢复再访问的条件，增加 synthetic provenance 标识 |

现有 `HostKVCache` 的 `_allocate_host_buffer()` 会按 `self.size * self.size_per_token` 检查物理可用 RAM 并创建 Host Tensor；其 `is_dummy` 可绕过分配，但 Host 数据读取/写入函数仍假定物理 Buffer 存在。`MLATokenToKVPoolHost(is_dummy=True)` 当前是 allocator-only 特殊分支，`load` 和 `backup` 在该分支显式 assert，因此**不能仅设置 `is_dummy=True` 就启动完整模拟**。`MambaPoolHost` 又直接自行分配各状态 Tensor，并未沿用这一 dummy 初始化路径；它需要额外改动。[S1–S4]

当前 `L2TransferEngine` 在 D2H Stream 上调用 `backup_from_device_all_layer_physical()`，在 H2D Stream 上逐层调用 `load_to_device_per_layer_physical()`，且每层经 `on_layer_done` 发出可等待事件。`HostPoolGroup` 允许多条 `L2Transfer`，因此 Mamba/MLA/MTP 可共用当前事件语义。[S5–S7]

## 4. 总体架构与数据流

```text
                  Real SGLang Scheduler / UnifiedRadixCache
                    ┌──────────────────────────────────┐
                    │ prefix match / host-only / evict │
                    └────────────────┬─────────────────┘
                                     │ logical host indices
                          HiCache / HybridCacheController
                                     │ CacheOperation / ACK
                        ┌────────────▼─────────────┐
                        │ VirtualL2TransferEngine  │
                        │ queue, chunk, stream, EVT│
                        └────────┬───────┬─────────┘
                                 │       │
                         D2H real DMA   H2D real DMA
                                 │       │
                   ┌─────────────▼───────▼────────────┐
                   │ Pinned Stage Pool (bounded RAM)  │
                   │ D2H sink / H2D synthetic source  │
                   └───────────────┬──────────────────┘
                                   │ physical stage slots reused after EVT
                    ┌──────────────▼─────────────────┐
                    │ Virtual Host Pool metadata      │
                    │ capacity / free / resident / gen │
                    │ *no per-slot full KV/SSM bytes*  │
                    └────────────────────────────────┘

       Real GPU L1 KV/SSM  ⇄  local real PCIe/NVLink-C2C DMA
       P GPU → D GPU      ⇄  original PD Mooncake/NIXL transfer
```

**Host slot 和 Stage slot 不能共用一个索引空间**。上层仍使用大容量逻辑 `host_indices`；仅在提交物理 DMA 时，把本 chunk 中每个逻辑 Host Slot 映射到小容量 `stage_indices`。多种缓存类型共用该规则，但数据布局、token/checkpoint 颗粒度、layer mapping 不应硬编码。

## 5. 关键类与接口提案

### 5.1 虚拟容量：`VirtualHostPool`

不需要为每个虚拟 Slot 创建 KV Tensor。仍向 Scheduler 提供与现有 HostPool 相容的 `alloc()`、`free()`、`available_size()`、`clear()`、`logical_size`、`logical_page_size`。新增只读 `PoolGeometry`，描述每种状态的 slot/page 计费；专门的 Page Allocator 避免超大 Token 数下的 `torch.arange()` 元数据增长。建议实现 **page-range/bitmap + generation**，逻辑 Slot ID 同原有方法兼容。

```python
@dataclass(frozen=True)
class VirtualPoolGeometry:
    pool_name: str                # full / swa / mamba / aux / draft ...
    bytes_per_logical_unit: int  # 按具体 Pool 定义，不保证等于 KV/token
    logical_page_size: int
    logical_capacity_pages: int
    layer_map: object            # 全局 layer -> Pool 本地 layer
    codec: str                   # MHA, MLA, MAMBA, HYBRID_SIDE

class VirtualHostPool:
    def alloc(self, logical_units: int) -> Tensor | None: ...
    def free(self, logical_indices: Tensor) -> int: ...
    def available_size(self) -> int: ...
    def prepare_transfer_indices(self, host_ids, device_ids, io_backend): ...
    # 数据操作由 VirtualL2TransferEngine + StageTransferAdapter 执行
```

**容量定义：** 每个 worker/rank 的总虚拟字节容量按 `Σ(pages_allocated × bytes_per_page)` 记账。如果存在 HostPoolGroup，子池分别限制；`--hicache-size` 的具体拆分仍沿用 SGLang 原本的 assembler 规则，不把 512GB 误乘成所有 rank 共享的 512GB。对于 DCP，要保留原版 widened logical indices 与 rank 局部 physical rows 的转换。

### 5.2 物理搬运：`StagePool`, `StageLease`, `StageTransferAdapter`

`StagePool` 在启动时创建少量 pinned Host Tensor，**包含 D2H sink 和 H2D source 两套资源**，分别支持 MHA K/V、MLA 压缩 KV、Mamba temporal + conv + extra checkpoint，以及 MTP draft / sidecar buffers。每个 Pool 的 stage tensor 沿用它原生的 page/layer layout，以复用现有 transfer kernel。

```python
@dataclass
class StageLease:
    stage_pool: object
    slot_ids: Tensor
    direction: str                 # D2H / H2D
    allocation_generation: int
    last_cuda_event: object | None
    released: bool = False

@dataclass
class StageChunk:
    logical_host_ids: Tensor       # 仅用于元数据与计量
    physical_stage_ids: Tensor     # ≤ stage pool page count
    device_indices: Tensor
    pool_name: str
    target_layers: list[int]
    byte_count: int

class StageTransferAdapter(Protocol):
    def bytes_for_chunk(self, chunk: StageChunk) -> int: ...
    def enqueue_d2h(self, chunk: StageChunk, stream) -> object: ...
    def enqueue_h2d_layer(self, chunk: StageChunk, layer_id: int, stream): ...
```

适配器是建议新增的抽象，不是现成 SGLang 公共接口。**不要**直接将逻辑 `host_indices` 传给物理小 Stage Pool 的 `kv_buffer` 索引，必须先做 Chunk Remapping。

### 5.3 `VirtualL2TransferEngine` 最小耦合方案

第一选择是在 `L2TransferEngine` 外包一层统一提交器：普通 Host Pool 沿用原分支，Virtual Host Pool 走 Stage Adapter。Controller 继续拿到单个聚合 `TransferCompletion`，对上层的 ACK、节点生命周期语义不改变。若现有 `submit_host_to_device()` 无法表达跨 chunk 的逐层调度，则最小扩展为 `submit_virtual_host_to_device(transfers, on_layer_done, ...)`。

**必须满足的接口契约：** (i) 返回的 Finish Event 在**所有 chunks 和全部参与 Pools**的 GPU DMA 完成后才触发；(ii) 对于 H2D 的每个全局 layer，先完成该 layer 所需全部 Pool 和 chunks 的写入操作，再 record `LayerLoadingEvent`；(iii) D2H 操作的 GPU 源 indices 被 lock 到 DMA 完成；(iv) Buffer lease 被保持到 CUDA 事件完成，而不是 Python 提交函数返回；(v) 异常、取消时能够完整回滚队列和页分配。

## 6. 真实 DMA：读写算法

### 6.1 D2H：真实复制、按完成时序丢弃

```text
Controller.write() → Virtual L2 logical slots allocated (PENDING_WRITE)
   → split by pool + physical staging budget + page boundary
   → acquire D2H StageLease (non-blocking or queue)
   → map device_indices → physical_stage_ids
   → wait source producer / L1 reuse gate
   → enqueue real device→pinned-host transfer (existing specialized kernel)
   → record Event(chunk_done) on D2H stream
   → when chunk_done: release StageLease; discard payload; next chunk
   → when all chunks done: completion ACK; mark virtual slots READY
```

当 D2H 尚未完成时，不能释放或覆写源 GPU Slot，也不能把该 L2 条目标记为可命中。`write_back` 只在备份 ACK 后允许有效 demotion；`write_through` / `write_through_selective` 同样需要两阶段提交。原本独立的 write queue/ack 逻辑应优先复用。

### 6.2 H2D：从稳定的合成缓冲执行真实拷贝

```text
L2 hit / init_load_back() → reserve actual GPU indices
   → acquire H2D StageLease (read-only seeded payload)
   → for each global layer:
         for each participating pool:
             for each chunk:
                 enqueue pinned-host→device copy into true GPU slot
         record LayerLoadingEvent(layer) on transfer stream
   → GPU forward stream waits for corresponding layer event
   → overall event finish → release leases; mark L1 RESIDENT
```

H2D 源缓冲需要**提前初始化并在所有使用者退出前保持只读**。可用固定零值或确定性伪随机值；首版推荐全零，以避免未初始化浮点数造成 NaN/Inf。`H2D source` 可由多个并发 Copy 只读共享，但其底层 Host allocation、tensor 对象以及所有 DMA 源地址均不得在 Event 前释放。覆盖性统计应确认每个被请求的 GPU bytes 均由实际 DMA 写入。

**DMA 真实性验证：** 单纯做一次 tiny H2D 并让计数器假报大字节数不合格。应按 `requested logical slots × true per-slot bytes` 提交真实 payload-size 的 DMA；大请求拆成多个 chunks，实际 PCIe bytes/transfer bytes 与系统 metrics 一致。通过 CUDA/Nsight/PCIe counter 与独立 staging microbenchmark 校验。

### 6.3 Stage buffer 容量与切块

设 B 为单实例 pinned RAM 预算，B = B_D2H + B_H2D + B_aux + B_metadata；每个方向设置 ring 中 N 个 slot，满足总物理 tensor size ≤ 对应预算。选择每个 chunk 的 page 数：

`chunk_pages = max(1, min(pages_remaining, floor(stage_free_bytes / bytes_per_page(pool))))`。

若单个 Mamba checkpoint / side state page 大于最大 stage slot，不能悄悄跳过；要么切分到 Tensor 子块（component/dtype/row 单位），要么启动时拒绝配置并报告最低预算。Stage Buffer 是有限资源，应由 `StageLease` + 队列公平性来回收：**D2H 与 H2D 不能共享同一块会被写入的地址**；同一 ring slot 只在其最后一个相关 CUDA Event 完成后重新可用。

### 6.4 分层 overlap 与防死锁

H2D 推荐按**layer-major → pool-major → chunk-major** 迭代：保证第 n 层所有 chunks 已提交后才 record 第 n 层 ready，接着 enqueue 第 n+1 层，保留 Forward/Transfer overlap。D2H 按 chunk-major 合并各层，复用 SGLang 现有一次备份所有层的路径。不能在 Scheduler 主线程中阻塞等 Stage 空闲；应让 transfer worker 异步排队、由完成事件驱动后续提交，避免阻塞 Forward。

如果 H2D 需要一个 stage slot 横跨全部层持续只读，优先采用静态、全生命周期有效的 H2D 模板池。对于 D2H，必须等待 Event 才能复用 sink。若受极小 stage 预算影响只能按层循环复用 D2H slot，可考虑增加 `enqueue_d2h_layer` 扩展接口；必须避免在同一 CUDA stream 上提交其未完成的物理覆盖。

## 7. 多缓存类型适配：一次覆盖所有目标

| 类型 | 逻辑单元与数据布局 | D2H/H2D 处理 | 额外不变量 |
|---|---|---|---|
| MHA/GQA | Token/Page × 层 × K/V head × head_dim；有不同 `layer_first`/`page_first` layout | 复用 `MHATokenToKVPoolHost` 的 JIT/direct Kernel，以 chunk 的 stage indices 替代 Host indices | TP 每 rank 的 KV head 量不同；字节统计按本 rank |
| MLA | Token/Page × layers × `(kv_lora_rank + qk_rope_head_dim)`，还可能有 indexer / FP8 scale / draft | 复用 `MLATokenToKVPoolHost` 搬运；按 `get_size_per_token()` / 实际 pool buffers 计量 | DCP rank 映射、layer sharding、draft、indexer side data 不得省略 |
| Gated DeltaNet/Mamba | **Checkpoint slot**（不是任意 token prefix 的完整逐 token KV）；各 layer 的 temporal recurrent state、conv state、extra checkpoint buffers | 物理 StagePool 为 temporal、每种 conv、所有 slot-state 单独建小 Tensor；H2D 在相应 layer 完成事件前恢复所需状态 | Mamba HostPool 当前 `page_size=1`；side state 目前在 `layer_id==0` 传输；保留此语义 |
| Hybrid | Full-Attn KV + SWA + Mamba 等多个子 Pool | 使用 `HostPoolGroup` 的 PoolEntry/Layer Mapper 调度多个独立 Stage 适配器 | 每个 component 的 lease/bytes/ACK 完整；共享 indices 时不得 double-free |
| MTP draft | Target + Draft Pool/打包 draft layers | 由 `L2Transfer.is_draft` 与 `layer_mapper` 路由，D2H/H2D 也计入 draft bytes | draft state 未加载就不能宣布相应 layer ready |

### 7.1 MHA/GQA

只保留虚拟 slot/eviction；所有物理 copy 发生在 `MHATokenToKVPoolHost` 兼容的小 `StagePool`。`get_page_buffer_meta()` 等返回原 Host Tensor 地址的接口不能暴露为逻辑 L2 地址。`kernel` 与 `direct` 的差异会影响真实 DMA 模型；在性能报告中必须单独标注 layout/backend。对 asymmetric K/V shape 使用 `get_mha_host_pool_cls()` 的选择结果构造 Stage 模板，而非假设 K/V 同形状。

### 7.2 MLA

`MLATokenToKVPoolHost.is_dummy` 只能借鉴其不分配内存的机制，不能直接恢复 KV；StageAdapter 应复用真实非 dummy MLA host pool 的接口，但物理容量设置为 staging 页数量。对于启用 DCP，遵循 `maybe_dcp_kernel_indices` 和当前 `host_transfer_translate`/logical-to-physical 的顺序；一个逻辑 host page 映射到多个 DCP row 时必须覆盖 rank 本地需传输的全部字节。Indexer、FP8 scale、MTP draft 及 MLA 的 rank replication 需要按当前已有 HostPool/PoolEntry 配置处理；不能简单按 `layers × (kv_lora_rank+rope)` 估算所有模型的最终 bytes。

### 7.3 Gated DeltaNet / Mamba

`MambaPoolHost` 的 `get_size_per_token()` 按 `[(conv + temporal) × num_mamba_layers] + side_state` 计算，而其逻辑 `size` 代表可保存的状态 slot 容量，并非能在任意 token 位置恢复完整历史。`temporal_buffer`、`conv_buffer[]`、`slot_state_buffers[]` 缺一不可，特别是 extra checkpoint 在代码中由 `_transfer_slot_states()` 负责，而且当前实现涉及 blocking CPU copies；虚拟分支应把这些复制调整到预分配 Stage Tensor，不得额外申请与虚拟容量成正比的 Host Tensor。由于不同 Gated DeltaNet/SSM 模型的 checkpoint/radix 语义可能不同，只支持 **上层真实 Scheduler 明确标注可恢复的 checkpoint**，不可凭任意 Prefix 命中制造不存在的 checkpoint。

### 7.4 Hybrid/混合模型

`HostPoolGroup.resolve_host_transfers()` 会按 PoolName 排序，为 side pools 分配 Host indices，失败时回滚；虚拟版本必须保存此原子性。一个主操作可能涉及多个 child leases：若任一子 Pool 的 DMA 失败，整个复合操作不能发布 `READY`，并按引用关系释放 `indices_from_pool` 的派生 Slot（不能 double-free）。`on_layer_done(layer)` 必须等待该层**所有** child Pool 的 DMA 提交完成与 CUDA Stream 事件记录。

## 8. PD Disaggregation：保持真实跨节点路径

当前文档说明 HiCache 的 L2 是 **instance-private**，并非跨节点共享；PD 的 Prefix/Decode 传输是独立的通信路径。故同一个 Virtual L2 总方案必须把三个通道分别建模：[S8–S10]

1. **Prefill worker 内：** P-GPU ↔ P-Virtual-L2，是真实本地 DMA，命中后的 P-GPU 数据可能为 synthetic。
2. **P→D handoff：** 仍走 SGLang 原有 Mooncake/NIXL 等 transfer backend，保留 bootstrap、destination indices、网络完成信号、重试/abort、异构 TP 映射。若上游 P 使用过 fake load，发往 D 的状态仍是 synthetic；报告要标识 provenance。
3. **Decode worker 内：** D-GPU ↔ D-Virtual-L2。支持 decode incremental offload、retraction backup 与 `DecodeHiCacheTransferMixin` 的 local restore；严格遵循在途 D2H 完成前不得释放 D 侧 GPU KV Slot 的约束。

### 8.1 PD 生命周期

```text
Request arrives at P
  → P prefix match (L1 / Virtual L2) → local H2D load if needed
  → real Prefill GPU execution
  → original PD handshake / remote KV transfer → D GPU slots
  → Decode real forward / tool calls
  → Decode incremental D2H via VirtualL2 (or cache eviction)
  → next turn: local decode-side restore OR new P prefill (route-defined)
```

**跨节点不共享逻辑 host slot ID。** 请求 ID、模型版本、pool kind、checkpoint epoch、prefix length、synthetic provenance 可用于 trace 对齐，但 P 和 D 的 L2 slots 属于各自 worker。若目标工作负载有 Decoder → Prefiller 的 L3 复用，则必须以 Shadow L3 方式额外模拟；不得把 synthetic tensor 当作真实可恢复 KV 存到共享存储。

### 8.2 Decode/offload、retraction 和 PD 边界条件

`DecodeKVCacheOffloadManager.offload_kv_cache()` 处理 Decode 增量 token，对 page-aligned 区间调用 `cache_controller.write()`。它的 `drain_before_retraction()` 会避免 D2H 未完成时错误回收 GPU 槽。VirtualL2 必须保留 `ongoing_offload` / `ack_write_queue` 等 drain 语义。`--disaggregation-decode-retraction-backup host_pool` 模式应单独做中断、取消、恢复测试。当前 PD 还有不同特性组合的校验与限制（如某些 `DCP`/HiSparse/role switch/异构 TP staging 组合），虚拟 L2 不应默认绕过现有校验；应复用原版参数检查，仅为已实测的组合逐项放行。[S11–S13]

## 9. 队列、带宽与真实 DMA 的关系

模式 B 的基础结果是“**在当前机器真实 DMA 带宽下**运行”。若目标模拟别的主机带宽，可选显式 **Bandwidth Pacing**，但只可降低有效吞吐，不能把慢硬件“加速”为更快物理链路：

`target_finish = max(now, serialized_direction_available_time) + fixed_latency + bytes / configured_effective_BW`。

完成 ACK 的时刻应为 `max(real_DMA_done, target_finish)`；GPU 转发使用相关 layer Event + 可选排队门控，不可等 CPU timer 结束却提前让模型读取未写完的 GPU KV。若目标目标主机的理论 BW 高于本机实测最大 DMA BW，标为 **不可直接物理复现**，只能输出基于模型的外推估计，不与模式 B 实测混报。

- D2H、H2D 各自一条或多条 copy queues；可配置双向同时进行与否，但实际由硬件约束。
- 采样排队延迟、Ring Lease Wait、DMA Kernel/PCIe Time、Event Wait、GPU Forward Stall 和 Completion ACK Time。
- 以每方向实际提交 bytes 和真实有效带宽为首要指标；**bytes 的口径包括所有 K/V、MLA indexer、Mamba side state、draft**。
- 可做两种 run：`native`（不附加人工 pacing）与 `paced`（加上 BW/固定延迟上界）。在图表上严格分开。

## 10. 配置提案（全部为建议新增项）

建议不要复用现有 `--hicache-io-backend` 的 `kernel/direct` 枚举：这是拷贝 Kernel 的选择，不应负责 Host 容量虚拟化。引入独立的 benchmark-only 参数：

```bash
# 提案示意：以下 virtual-l2-* 均非上游已有参数
python -m sglang.launch_server \
  --model-path MODEL_PATH \
  --enable-hierarchical-cache \
  --hicache-write-policy write_back \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --virtual-l2-enable \
  --virtual-l2-capacity-gb 512 \
  --virtual-l2-staging-gb 2 \
  --virtual-l2-stage-ring-entries 4 \
  --virtual-l2-data-pattern zeros \
  --virtual-l2-dma-mode native \
  --virtual-l2-disable-real-l3
```

推荐所有配置进入独立 dataclass，而非全塞 ServerArgs；新增 `virtual_l2_allow_pd` 明确标记性能实验模式，并在日志、metrics、输出 JSON 中打印 `synthetic_cache_data=true`。原生 `--hicache-size` 仍用于常规真实 L2，不建议同时设置两个相互冲突的容量真值；应由虚拟工厂将 virtual capacity 映射到现有 host pool sizing 分配，并显式打印 per-rank 计算结果。

## 11. 代码改动与建议目录

```text
python/sglang/srt/mem_cache/
  virtual_l2/
    __init__.py
    config.py                 # VirtualL2Config, guards
    geometry.py               # per pool / model / rank sizing
    allocator.py              # page bitmap/ranges/generation
    pool.py                   # VirtualHostPool, virtual HostPoolGroup adapters
    staging.py                # pinned StagePool, rings, StageLease
    transfer.py               # chunk submission, DMA, events, ACK fan-in
    adapters/
      mha.py                  # MHA / asymmetric / SWA
      mla.py                  # MLA / DCP / indexer / draft
      mamba.py                # temporal + conv + extra side state
      hybrid.py               # composite PoolTransfer / mappings
    metrics.py                # real bytes, waits, event timing
    provenance.py             # synthetic state trace propagation
```

现有文件的最小 patch：`hybrid_pool_assembler.py` + `kv_cache_builder.py` 增加 Virtual L2 工厂；`L2TransferEngine` 增加虚拟 submit 路径；`cache_controller.py` 增加可能的分块 ACK fan-in 与错误清理；Decode offload/retraction 复用同一 host pool 工厂。应尽量不动 `UnifiedRadixCache` 的 prefix、eviction 决策，不能跳过 `HostPoolGroup` 的 side-pool 回滚。

## 12. 并发状态机和关键不变量

每个逻辑 Cache Page 记录：`FREE → RESERVED → PENDING_D2H → READY_L2 → PENDING_H2D → READY_L1`，其中 `READY_L2` 可以与 L1 resident 并存；失败沿路径转向 `CANCEL_PENDING → INVALID → FREE`，且须按引用数/锁释放。状态转换只由明确的 CUDA Completion 与 controller ACK 推进，**不能由提交成功直接推进到 READY**。

**六条强制不变量：**

1. `logical_allocated_bytes ≤ virtual_capacity_bytes` 且 `physical_pinned_bytes ≤ staging_budget + measured_metadata_overhead`（metadata 单独报）。
2. 一个 D2H Stage Lease 在最后一次读取/写入完成 Event 前不得重分配；所有 GPU source slots 在 Event 完成前不得被 evict/retract。
3. 任一 layer 的 ready Event 必须发生在该 layer 所需全部 H2D CUDA 操作被提交到同一依赖链后；Compute Stream 必须 wait 它。
4. 任一 Hybrid ACK 必须代表该复合请求的所有 child pools 成功，而不是仅 anchor pool 完成。
5. 任何 Host-only Prefix 在 H2D 之前都必须有有效的 `READY_L2` 元数据；`FREE/INVALID/PENDING_D2H` 不可命中。
6. 开启 PD 时 local L2 与 P→D 不共享 slot 地址；取消、节点失败或 timeout 必须同时清理本地 lease 与远程 transfer 事务。

## 13. 实验方法与评估指标

### 13.1 场景矩阵

| 维度 | 配置/实验 |
|---|---|
| Model | 常规 MHA/GQA、MLA、Gated DeltaNet/SSM Hybrid、MTP on/off |
| Serving | 单机单卡、TP、DP-Attn（适用时）、PD P/D 独立 worker |
| L2 容量 | 物理可容纳小基线 / 64 / 128 / 256 / 512 / 1024 GB（per-rank 记账） |
| Staging RAM | 0.25 / 0.5 / 1 / 2 / 4 GB（适配最小 checkpoint 大小） |
| Cache | `write_back` / `write_through` / `write_through_selective` |
| Local H2D | 仅 L1 命中、L2 全命中、部分命中、Host-only 冷恢复、反复 eviction |
| Hybrid | Full+SWA、Full+Mamba、MLA+indexer、Conv-only、extra state、MTP draft |
| PD | P-only HiCache、P+D local Virtual L2、D incremental offload、D retraction/restore、Abort/Timeout |
| Trace | 共享 prefix 的多轮 Agent、tool gap、多请求并发、compaction 前后 prefix 变化 |

### 13.2 报告的核心指标

- Throughput：finished requests/s、tokens/s；TTFT、TPOT、E2E latency、queue time、GPU idle time。
- Cache：L1/L2 hit tokens、resident / evict tokens、logical L2 occupancy、page fragmentation、write success/drop、host-only prefixes、Mamba checkpoint hit。
- DMA：真实 D2H/H2D bytes、kernel 提交次数、有效 GB/s、direction queue wait、lease wait、completion latency、layer-overlap 时间。
- 内存：RSS、pinned allocation peak、stage pool peak、GPU HBM、allocator metadata bytes；每个 rank 分别报告。
- PD：P→D 真正网络 transferred bytes/latency、bootstrap wait、D waiting、retraction count、decode offload queue、每角色的 L2 hit 与 DMA。
- 真实性：`actual_dma_bytes / predicted_dma_bytes`、staging slot reuse violations（必须为 0）、premature ACK（必须为 0）。

### 13.3 有效对照设计

**A 组：** 小容量真实 L2，使用真实数据；**B 组：** 同样逻辑容量但 Virtual L2，用来计算 DMA/调度相对误差；**C 组：** 仅扩大 Virtual L2 到目标容量并测系统表现；**D 组（可选）：** 离线 Simulator，验证缓存策略趋势。只有 A vs B 可在硬件真实可容纳的容量下直接对照；C 的更大容量只是对目标硬件的模拟预测。对于数值敏感的 MoE、Mamba 模型，固定 prompt trace / 固定 decode token 个数 / 强制外部 tool timings，避免 synthetic 内容诱发的 EOS 与执行路径差异。即使强制输出 Token，也不能保证 data-dependent kernels 与 MoE expert 分配完全一致，应报告偏差。

## 14. 测试与验收清单

**单元测试：** Page allocator 的分配/释放/双重释放/跨页对齐；每 Pool geometry bytes；HostPoolGroup 原子回滚；DCP/TP ID 映射；StageLease generation、防 double release；切块 bytes 守恒；高并发 D2H/H2D；失败/取消超时不泄漏 lease；Mamba extra checkpoint side-state 全覆盖。

**集成测试：** 先用小内存下的 Real L2 与 Virtual L2 对照；对每个模型禁用 L3；触发 Host-only cache hit、eviction、prefill/load overlap、MTP、Mamba checkpoint；高并发反复触发 GPU Slot retraction；保存 Nsight Systems traces 并检查 Event 因果链。

**PD 测试：** P/D 分别开启或不开启 Virtual L2；真实 network transfer bytes 与事件；Decode offload 后 retraction 和恢复；Abort before/after DMA；转移失败的 GPU/Host Slot 回收；不同 TP 拓扑下使用当前上游合法的 backend；L3 仅 Shadow Namespace 时才能额外跑存储复用。

**通过阈值提案：** (a) CUDA race/memory sanitizer 无未定义访问，(b) byte accounting 偏差在严格定义口径后为零，(c) pinned RAM 不随虚拟容量增长，(d) 虚拟化前后（**相同且真实可容纳的容量**）调度命中结果一致，(e) 将未实现的组合显式 fail-fast，不能无声回退成跳过 DMA。

## 15. 项目落地分工与交付顺序（所有架构都在同一版本范围内）

建议将同一开发版本拆为可独立测试的提交，而不是把 MLA/Mamba/PD 作为未来可选功能：

1. **基础设施提交：** Config、Virtual Page Allocator、StageLease/Ring、DMA Engine、metrics、通用故障清理。
2. **模型 Adapter 提交：** MHA/GQA → MLA/DCP/indexer/draft → Mamba/DeltaNet/extra checkpoint → Hybrid Pool；这些 adapter 均列入最终交付验收。
3. **PD 提交：** Prefill/Decode 双角色接入、Decode incremental offload、retraction、local restore、trace provenance；保留真实原生 P→D 通道。
4. **评估提交：** 实验矩阵、真实小 L2 对照、stage size sweep、DMA profile、Agent trace replay、完整的 README 与 reproducibility manifest。

发布判定是**四个提交类别全部落地**，不把第三项安排成所谓“未来第三版”。

## 16. 风险与对应措施

| 风险 | 后果 | 应对 |
|---|---|---|
| Stage 容量小于最大 checkpoint/page | D2H/H2D 无法提交 | 启动时计算最小 stage bytes；按 tensor component 切块或 fail-fast |
| 错把 `is_dummy` 作为可传输 Host Pool | `kv_buffer=None`、assert | 单独 Virtual+Stage 组合类，全部传输仅针对 Stage Tensor |
| Stage 提前复用 | DMA 数据竞态、崩溃 | 事件驱动 release、generation id、压力测试 |
| 只拷少量 bytes 却计全量 | 吞吐虚高 | 设备侧 profiler 统计与 bytes accounting 交叉核对 |
| Hybrid/MTP 部分状态未恢复 | 结果不可信、GPU 未初始化 | 显式 PoolEntry 迭代、复合 ACK、逐 Pool 覆盖测试 |
| Mamba Side State 通过 blocking CPU copy 绕过 stage | 隐式巨额 RAM / 时序不准 | 显式将 extra_checkpoint_buffers 纳入 StageAdapter 与事件链 |
| Fake KV 改变 MoE/SSM 执行轨迹 | 不同工作负载，吞吐偏差 | 外部 trace 驱动、固定输出长度、单列 validation 偏差 |
| 旧/新 PD 协议和本地虚拟池索引混用 | Remote restore 错误 | P→D 保留真数据结构、接口 fail-fast、严格节点私有 Host IDs |
| 实际 DMA 本身比目标机器更慢 | 无法模拟更快主机 | 只报告 `native` 性能与模型外推结果，禁止伪装成测量值 |
| 虚拟数据污染 L3/生产缓存 | 后续真实请求不正确 | 默认禁止生产 L3，Shadow namespace + 显式 synthetic 标记 |

## 17. 源码与设计依据

- **[S1]** [`pool_host/base.py`](https://github.com/sgl-project/sglang/blob/06934575b9eb0620832e842facc0591b5a3e9b3f/python/sglang/srt/mem_cache/pool_host/base.py)：`HostKVCache` 构造、`is_dummy`、Host 分配、Slot API、物理拷贝 hooks。
- **[S2]** [`pool_host/mha.py`](https://github.com/sgl-project/sglang/blob/06934575b9eb0620832e842facc0591b5a3e9b3f/python/sglang/srt/mem_cache/pool_host/mha.py)：MHA/GQA layouts、Stage Write-back、data transfer。
- **[S3]** [`pool_host/mla.py`](https://github.com/sgl-project/sglang/blob/06934575b9eb0620832e842facc0591b5a3e9b3f/python/sglang/srt/mem_cache/pool_host/mla.py)：MLA Dummy 分支、DCP、KV+Indexer layouts、MTP draft。
- **[S4]** [`pool_host/mamba.py`](https://github.com/sgl-project/sglang/blob/06934575b9eb0620832e842facc0591b5a3e9b3f/python/sglang/srt/mem_cache/pool_host/mamba.py)：MambaPoolHost、conv/temporal/slot state、拷贝与分配。
- **[S5]** [`l2_transfer.py`](https://github.com/sgl-project/sglang/blob/06934575b9eb0620832e842facc0591b5a3e9b3f/python/sglang/srt/mem_cache/l2_transfer.py)：L2Transfer、异步 submission、per-layer loading 与 streams。
- **[S6]** [`managers/cache_controller.py`](https://github.com/sgl-project/sglang/blob/06934575b9eb0620832e842facc0591b5a3e9b3f/python/sglang/srt/managers/cache_controller.py)：CacheOperation、HiCacheAck、LayerLoadingEvent、Write/Load queue。
- **[S7]** [`pool_host/group.py`](https://github.com/sgl-project/sglang/blob/06934575b9eb0620832e842facc0591b5a3e9b3f/python/sglang/srt/mem_cache/pool_host/group.py)：HostPoolGroup、PoolEntry、resolve_host_transfers 原子回滚。
- **[S8]** [HiCache Design](https://docs.sglang.ai/advanced_features/hicache_design.html)：L1/L2 私有性、L3 范围、Write Policy、DMA/Layer overlap。
- **[S9]** [HiCache Best Practices](https://docs.sglang.ai/advanced_features/hicache_best_practices.html)：HiCache/PD 组合及 Decode 侧 offload。
- **[S10]** [PD Disaggregation](https://docs.sglang.ai/advanced_features/pd_disaggregation.html)：真实 Prefill↔Decode 传输、不同 TP 的 staging 及限制。
- **[S11]** [`decode_kvcache_offload_manager.py`](https://github.com/sgl-project/sglang/blob/06934575b9eb0620832e842facc0591b5a3e9b3f/python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py)：Decode incremental offload 和 retraction 同步。
- **[S12]** [`decode_hicache_mixin.py`](https://github.com/sgl-project/sglang/blob/06934575b9eb0620832e842facc0591b5a3e9b3f/python/sglang/srt/disaggregation/decode_hicache_mixin.py)：Decode local restore gating。
- **[S13]** [`arg_groups/hicache_hook.py`](https://github.com/sgl-project/sglang/blob/06934575b9eb0620832e842facc0591b5a3e9b3f/python/sglang/srt/arg_groups/hicache_hook.py) 与 [`pd_disaggregation_hook.py`](https://github.com/sgl-project/sglang/blob/06934575b9eb0620832e842facc0591b5a3e9b3f/python/sglang/srt/arg_groups/pd_disaggregation_hook.py)：现有参数合法组合限制。

---

**设计决策：** 采用逻辑缓存和物理 DMA 解耦；小 pinned Stage Buffer 负责真实数据搬运，Virtual Pool 负责巨大容量；保留真实调度和 PD 网络路径。所有模型架构同批纳入功能范围，所有输出一律标记为 **synthetic cache / performance-only**。
