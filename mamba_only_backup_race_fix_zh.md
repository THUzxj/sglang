# Mamba-only Host 写回与暂停保护竞态修复

## 摘要

本文记录一个 Unified Radix Cache 暂停请求失败问题：节点的 Full KV 已经备份到 host，但该节点后来出现的 Mamba device checkpoint 没有对应的 Mamba host checkpoint。暂停保护要求两者都可从 host 恢复，因此在这种状态下拒绝保护节点。

修复的核心是在 Full KV 已有 host 副本时，允许只写回缺失的 Mamba 状态；不重新分配、不覆盖已有 Full host slot，并在写入完成后才发布新的 Mamba host 指针。暂停保护还需要锁住相关 host 节点，并让锁 lease 在 radix 节点拆分时继续覆盖新旧节点。

## 现象与诊断

复现日志来自：

```text
/data/xjzhang/context_engineering_serving/serving_system/results/kernel_bench_3turn_compact_turn1_decode1000_ce2000_tool10/para-freq4_pair_scheduler_radix_evictable_aiperf_20261021_210012/client/para-freq4_pair_scheduler/aps_5_nreq_200/server/server/decode/server.log
```

暂停保护失败时，诊断字段呈现如下状态：

| 状态 | 值 | 含义 |
| --- | --- | --- |
| `full_host` | `True` | Full KV 已有 host 副本 |
| `mamba_device` | `True` | 当前节点有 Mamba device checkpoint |
| `mamba_host` | `False` | 缺少可恢复的 Mamba host checkpoint |
| `pending_write` | `None` | 当时没有登记中的写回 |
| `ongoing_write_ids` | `[]` | 当时没有等待 ACK 的写回 |

同一请求的前序缓存节点曾完成过备份，但当时 `mamba_device=False`，因此仅有 Full KV 被写回。后续插入/暂停路径使 Mamba device 状态出现在节点上；再次备份时，通用备份逻辑依据 Full 的 `backuped` 状态跳过整个节点。结果是 Full 与 Mamba host 状态不一致，且没有异步写回可以等待。

这不是单纯的“暂停时等待太早”问题：没有 pending write 或 ACK 可等待。根因是备份跳过条件只代表 Full KV 已备份，却被当作所有组件都已备份。

## 修复设计

### 1. 只补写 Mamba，不动 Full slot

在 `python/sglang/srt/mem_cache/unified_radix_cache.py` 的已备份节点分支中，检查 Mamba device/host 状态。若 Mamba device checkpoint 存在而 Mamba host checkpoint 缺失，调用 `_execute_mamba_only_backup()`。

该写回构造零长度 Full 索引，只把 Mamba transfer 和其依赖的 sidecar transfer 交给 Hybrid Cache Controller。Controller 仍负责为组件分配 host slot 并执行传输，但 Full KV 的数据长度为零，因此不会重写现有 Full host 内容。若组件 host slot 分配失败，controller 的 transfer allocation 路径会回滚此次成功分配的组件 host slot，且不会提交 Mamba host 指针。

### 2. ACK 后发布 Mamba host 指针

Mamba-only 路径保留待提交的 Mamba transfer 描述，先提交写操作并登记 write-through pending 状态，然后通过 `writing_check(write_back=True)` 等待写入完成。只有同步等待成功返回后，才调用 Mamba component 的 `commit_hicache_transfer(BACKUP_HOST)` 发布 `host_value`。

这让暂停保护不会把“已分配但尚未写完”的 host slot 当作有效 checkpoint。Full host slot 在整个过程中保持原样。源节点在写回期间通过 lock ref 固定，降低 Mamba device slot 被并发回收的风险。

### 3. 暂停保护与 lease 生命周期

暂停保护先确保 Full backup 存在；若该节点有 Mamba device 状态但没有 Mamba host 状态，则同步执行 Mamba-only 写回。成功后再对 Full 节点和可用的 Mamba host 节点获取 host lock。

暂停期间 radix tree 仍可能发生 split。若只锁原节点，split 后承载 host 数据的新 parent 可能不受保护。因此 `unified_tree_core.py` 注册 paused-host-lock lease，并在 `_split_node()` 时把相应锁扩展到新 parent；lease 的释放路径统一撤销所有关联节点的锁。scheduler 和 decode 清理路径负责在请求生命周期结束时释放 lease。

## 代码位置

- `python/sglang/srt/mem_cache/unified_radix_cache.py`：已备份节点的缺失组件检查、Mamba-only 写回、暂停保护的补写与状态诊断。
- `python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py`：paused host-lock lease 注册、释放及节点 split 时的锁扩展。
- `python/sglang/srt/managers/scheduler.py`：scheduler 请求清理时释放 paused host lock。
- `python/sglang/srt/disaggregation/decode.py`：decode 请求清理时释放 paused host lock。
- `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`：Mamba-only 写回 ACK 提交顺序、Full/Mamba host 锁配对、split 后 lease 完整释放的测试。

## 回归验证

已在 `sglang_v0517_early_compaction_qwen35` 容器中运行：

```bash
docker exec -w /data/xjzhang/context_engineering_serving/references/sglang_dev \
  -e PYTHONPATH=/data/xjzhang/context_engineering_serving/references/sglang_dev/python \
  sglang_v0517_early_compaction_qwen35 \
  python -m unittest discover -s test/registered/unit/mem_cache \
  -p test_unified_radix_cache_unittest.py -k TestPausedCacheProtection
```

结果：3 项测试通过（`Ran 3 tests ... OK`）。此外，修改过的 Python 文件通过 AST 解析，`git diff --check` 通过。

## 尚未完成的验证

`test_simple_para.sh` 的完整 AIPerf/服务端实验尚未验证成功。运行前的资源检查发现 GPU 0–3 均已有其他 SGLang scheduler 进程占用，显存约 34–35 GiB / 40 GiB；为避免中断其他任务，没有启动脚本。因此当前结论是“针对性单元测试通过”，不能据此声称 200 请求的端到端实验通过。

资源允许后，从仓库根目录运行：

```bash
bash test_simple_para.sh
```

验收应确认脚本 exit code 为 0、200 请求正常完成，且 decode server 日志不再出现 Mamba backup missing / paused-node protection failed 或相关 scheduler traceback。运行时应在启动约一分钟后检查早期失败，之后约每五分钟监控，并将本次日志/结果路径记录在实验记录中。

## 后续排查要点

若仍失败，优先检查以下事件顺序和字段：

1. `Paused-node protection begin`：确认 `full_host`、`mamba_device`、`mamba_host`、`pending_write` 与 `ongoing_write_ids`。
2. `Mamba-only backup submitted` / allocation failed：确认 Mamba device checkpoint 是否存在、host pool 是否有容量，以及 Full host slot 是否仍保留。
3. write-through ACK：确认 ACK 等待成功后才出现 Mamba host commit；失败时不要将未完成 transfer 的 host slot 发布为有效状态。
4. `Paused-node protection established` 与 lease 清理：确认锁覆盖 Full/Mamba host 节点，节点 split 后新增 parent 也受保护，并在成功、失败、取消路径都释放 lease。
5. 检查完整 server traceback 和同一请求前后的节点状态，区分容量不足、device checkpoint 已回收、transfer/ACK 失败与节点 split/锁释放竞态。
