可以。你的目标本质上不是现在 SGLang 的 `fake`，而是一个 **metadata-only PD backend**：

```text
真实：
Decode ──dst page metadata──> Prefill
Decode <===== KV + aux ====== Prefill

你要的：
Decode ──dst page metadata──> Prefill
Decode <── first-token/aux ── Prefill
           X KV payload
```

我建议新增 `metadata_only` / `fake_connected` backend，**不要修改现有 `fake`**。现有 `fake` 是 decode-only benchmark 的特殊语义：`FakeKVManager` 只继承 `BaseKVManager`，没有真实 bootstrap/ZMQ control plane；sender/receiver 也只是本地状态翻转。

## 0. 当前本地代码的推荐实现

下面这一节按当前 `references/sglang_dev` 代码来写，优先级高于后面较早的草案内容。后面的草案里有一些点仍然有价值，但两处建议需要修正：

- 不建议新增独立的 `b"META_ONLY"` wire format。当前 `CommonKVManager._start_bootstrap_thread()` 已经能接收 `GUARD + TransferInfo.from_zmq()` 格式的 decode metadata，沿用它可以少改 prefill bootstrap 逻辑，也能复用 TP/CP/PP rank mapping、dummy rank、decode prefix len、abort notification。
- 不建议只手工回传 `output_id` 和 `cached_tokens[0:7]`。当前 `MetadataBuffers` 已经承载 output id、cached token 统计、logprob、sampling mask、spec/topk hidden states、DSA topk 以及 bootstrap_room。metadata-only backend 应该把 prefill 侧 aux metadata slot 的完整 buffer 内容复制到 decode 侧对应 aux slot，再写入 `bootstrap_room` gate。

### 0.1 目标语义

`metadata_only` backend 应该是一个真实 PD control-plane backend，而不是 `fake` 的另一个名字：

```text
Decode:
  1. 查询 bootstrap server，拿到 prefill rank_ip/rank_port。
  2. 正常预分配 decode KV pages。
  3. 发送 dst page indices、dst aux slot、state indices、decode_prefix_len。
  4. 等待 prefill 回传 aux metadata。
  5. 不等待任何真实 KV payload。

Prefill:
  1. 正常注册到 bootstrap server。
  2. 正常接收 decode metadata，room 收齐后进入 WaitingForInput。
  3. 正常执行 prefill forward。
  4. `send(kv_indices, ...)` 不做 KV copy，只记录 metric/progress。
  5. 最后一个 chunk 之后，把 prefill aux metadata slot 发回 decode。
```

这个 backend 保留：

- bootstrap HTTP server 与 prefill rank registration
- decode heartbeat 与 node failure tracking
- decode KV preallocation 和 page table construction
- decode 到 prefill 的 metadata serialization/ZMQ 传输
- prefill queue、batching、chunked prefill、first-token metadata
- decode metadata gate，即 `metadata_buffers.bootstrap_room[idx]` 到达后才允许请求转入 decode

它去掉：

- NIXL agent 初始化
- NIXL memory registration
- KV/state tensor RDMA copy
- NIXL notification/progress engine

### 0.2 和 NIXL backend 的关系

NIXL 当前大致分三段：

```text
Decode CommonKVReceiver:
  /route 查询 prefill topology
  -> 注册 decode KV/aux/state base pointers
  -> send_metadata() 发送 room、dst pages、aux_index、state indices、decode_prefix_len

Prefill NixlKVManager:
  _start_bootstrap_thread() 接收 KVArgsRegisterInfo / TransferInfo
  -> 收齐 required_dst_info_num 后 room = WaitingForInput

Prefill NixlKVSender:
  send() 把每个 chunk 加入 transfer queue
  -> worker 真实执行 KV/state/aux transfer
  -> NIXL notification 告诉 decode 已完成
```

`metadata_only` 应该复用第一段和第二段，只替换第三段。也就是说，metadata-only 更像是 “NIXL without data plane”，不是 “fake with bootstrap”。

关键差异：

| 行为 | NIXL | metadata_only |
| --- | --- | --- |
| Bootstrap server | `CommonKVBootstrapServer` | `CommonKVBootstrapServer` |
| Prefill rank registration | `CommonKVManager.register_to_bootstrap()` | 同 NIXL |
| Decode metadata | ZMQ `GUARD` multipart | 同 NIXL |
| Decode KV pointer registration | 需要，供 RDMA 写入 | 可以发送空 agent metadata，KV ptr 可以保留用于格式兼容 |
| Prefill KV chunk | enqueue NIXL transfer | no-op，只记录 progress |
| Aux metadata | NIXL 写 decode aux ptr，并发 notification | ZMQ 发送完整 aux slot bytes，decode 写本地 buffer |
| Decode completion | `check_transfer_done()` 收齐 KV/state/aux notif | aux bytes 写入且 room gate 写入后 `KVPoll.Success` |

### 0.3 建议新增文件

新增目录：

```text
python/sglang/srt/disaggregation/metadata_only/
  __init__.py
  conn.py
```

`conn.py` 里的类：

```python
class MetadataOnlyKVManager(CommonKVManager):
    def set_metadata_buffers(self, metadata_buffers): ...


class MetadataOnlyKVSender(CommonKVSender):
    def send(self, kv_indices, state_indices=None, num_kv_tokens=None): ...
    def poll(self) -> KVPoll: ...


class MetadataOnlyKVReceiver(CommonKVReceiver):
    def send_metadata(self, kv_indices, aux_index=None, state_indices=None, decode_prefix_len=None): ...
    def poll(self) -> KVPoll: ...


class MetadataOnlyKVBootstrapServer(CommonKVBootstrapServer):
    pass
```

### 0.4 Manager

`MetadataOnlyKVManager` 继承 `CommonKVManager`，不要继承 `FakeKVManager`。

Prefill 模式下，`CommonKVManager.__init__()` 已经：

- 绑定 `self.server_socket`
- 注册到 bootstrap server
- 初始化 `transfer_infos`

metadata-only manager 需要自己启动一个轻量 `_start_bootstrap_thread()`，逻辑可以从 NIXL 精简出来，只解析 decode metadata，不做 NIXL agent 或 pointer registration。

Decode 模式下，它已经：

- 初始化 `connection_pool`
- 初始化 `required_prefill_response_num_table`
- 初始化 `prefill_response_tracker`

metadata-only manager 需要自己启动 decode aux listener 和 heartbeat checker。

metadata-only 需要补的只是 aux metadata buffer 的 Python tensor 引用：

```python
class MetadataOnlyKVManager(CommonKVManager):
    def set_metadata_buffers(self, metadata_buffers):
        self.metadata_buffers = metadata_buffers
```

然后在 prefill 和 decode 的 `_init_kv_manager()` 创建 manager 后调用：

```python
if hasattr(kv_manager, "set_metadata_buffers"):
    kv_manager.set_metadata_buffers(self.metadata_buffers)
```

### 0.5 Decode send_metadata

`MetadataOnlyKVReceiver.send_metadata()` 可以从 `NixlKVReceiver.send_metadata()` 精简而来：

- 若 `bootstrap_infos is None`，标记 room failed。
- 遍历每个 `bootstrap_info`。
- 发送 `GUARD` 格式的 TransferInfo 消息：

```text
[
  GUARD,
  room,
  decode_local_ip,
  decode_rank_port,
  decode_agent_name,
  dst_kv_indices int32 bytes,
  dst_aux_index,
  required_dst_info_num,
  packed_state_indices,
  decode_prefix_len,
  is_dummy,
]
```

这里 `decode_agent_name` 不需要是真实 NIXL agent 名，可以用稳定字符串，例如：

```python
self.kv_mgr.agent_name
```

或者在 manager 初始化时生成：

```python
self.agent_name = f"metadata_only_{uuid.uuid4()}"
```

注意：为了让 `CommonKVManager._start_bootstrap_thread()` 不需要改，frame 位置应保持和 NIXL `TransferInfo.from_zmq()` 一致。

### 0.6 Decode KVArgs registration

`CommonKVReceiver._setup_bootstrap_infos()` 会调用 `_register_kv_args()`。metadata-only 有两种选择：

1. 最小实现：override `_register_kv_args()`，直接返回 `True`。因为 metadata-only 不需要 prefill 知道 decode KV/aux pointer。
2. 兼容实现：发送 `room == "None"` 的注册消息，但 agent metadata 为空。这样 prefill 可以保存 `KVArgsRegisterInfo`，方便未来扩展和 debug。

第一版建议选 1，减少依赖 NIXL 的 `KVArgsRegisterInfo`、`pack_int_lists`、pointer layout 和 agent metadata。

### 0.7 Prefill Sender

`MetadataOnlyKVSender.send()` 应该使用 `CommonKVSender._prepare_send_indices()`，这样 chunked prefill、CP dummy rank、`curr_idx`、last chunk 判断都和真实 backend 一致。

伪代码：

```python
def send(self, kv_indices, state_indices=None, num_kv_tokens=None):
    kv_indices, index_slice, is_last_chunk, should_skip = (
        self._prepare_send_indices(kv_indices, state_indices)
    )
    if should_skip:
        return

    self._record_transfer_indices(kv_indices, state_indices)
    if self._transfer_start_time is None:
        self._transfer_start_time = time.perf_counter()

    if is_last_chunk:
        self._send_aux_metadata()
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Success)
```

`poll()`：

```python
def poll(self):
    status = self.kv_mgr.check_status(self.bootstrap_room)
    if status == KVPoll.Bootstrapping:
        timeout = self._check_bootstrap_timeout()
        if timeout is not None:
            return timeout
    return status
```

`__init__()` 里要记录 `init_time = time.time()`，这样 bootstrap timeout 仍然有效。

### 0.8 Prefill aux metadata 回传

prefill 侧在最后一个 chunk 时，`prefill.py` 会先调用：

```python
self.disagg_metadata_buffers.set_buf(req)
```

再调用 sender 的 `send()`。所以 sender 可以读取 `self.kv_mgr.metadata_buffers` 的 `self.aux_index` slot。

推荐协议：

```text
[
  b"METADATA_ONLY_AUX",
  room,
  dst_aux_index,
  pp_rank,
  aux_slot_0,
  aux_slot_1,
  ...
]
```

其中 `aux_slot_i` 来自 `MetadataBuffers.get_buf(src_idx)` 返回的每个 tensor clone：

```python
payload = self.kv_mgr.metadata_buffers.get_buf(self.aux_index)
```

发送时把每个 tensor转成 CPU contiguous bytes，并带上简单 dtype/shape header，或者更简单地按固定顺序逐字段发送，因为 decode 侧的 `MetadataBuffers` shape 与 prefill 侧一致。第一版可以用固定顺序，代码短且和 SGLang 当前 buffer 定义绑定。

Decode 侧收到后必须按顺序：

1. 写入 decode metadata buffers 的 `dst_aux_index` slot。
2. 写入 `bootstrap_room[dst_aux_index, 0] = room`。
3. `update_status(room, KVPoll.Success)`。

顺序不能反，因为 decode 的 metadata gate 会检查：

```python
metadata_buffers.bootstrap_room[decode_req.metadata_buffer_index, 0]
```

如果先标记 `Success`，poll 线程可能在 metadata 写入前推进请求。

### 0.9 不要改 `_is_fake_transfer()`

不要把 `metadata_only` 加进 `_is_fake_transfer()`。

`_is_fake_transfer()` 的用途是让真正的 fake-bootstrap 请求绕过一些真实 PD 逻辑，例如 metadata gate、prefill dp rank resolution、decode-side hisparse direct transfer guard。metadata-only backend 的目的正好相反：它要验证这些 control-plane 和 scheduler 状态机路径，因此必须被当作真实 transfer backend。

### 0.10 server_args 与 backend 注册

需要改：

```text
python/sglang/srt/server_args.py
python/sglang/srt/disaggregation/utils.py
```

`server_args.py` 中所有 `disaggregation_transfer_backend` choices 都要包含：

```text
metadata_only
```

`utils.py` 中：

```python
class TransferBackend(Enum):
    METADATA_ONLY = "metadata_only"
```

`get_kv_class()` 中返回：

```python
{
    KVClassType.KVARGS: KVArgs,
    KVClassType.MANAGER: MetadataOnlyKVManager,
    KVClassType.SENDER: MetadataOnlyKVSender,
    KVClassType.RECEIVER: MetadataOnlyKVReceiver,
    KVClassType.BOOTSTRAP_SERVER: MetadataOnlyKVBootstrapServer,
}
```

### 0.11 第一版限制

第一版建议显式限制和测试：

```text
P TP == D TP
CP = 1
PP = 1
dense attention
不开 speculative decoding
不开 Mamba/SWA/state transfer
不开 staging buffer
不开 decode radix cache
```

原因不是这些场景一定不能支持，而是它们会引入额外状态：多 PP rank completion、state indices、staging watermark、decode prefix hit 的 no-KV last chunk notification。metadata-only 的第一版目标是得到可靠的 zero-KV-data-plane baseline。

### 0.12 测试路径

先做轻量验证：

```bash
python3 -m compileall \
  python/sglang/srt/disaggregation/metadata_only \
  python/sglang/srt/disaggregation/utils.py \
  python/sglang/srt/server_args.py
```

然后在 docker 里用 `lmsysorg/sglang:v0.5.17` 和 host 上的 customized repo：

```text
/data/xjzhang/context_engineering_serving/references/sglang_dev
```

跑 `serving_system/scripts/run_qwen36_27B_aps_sweep.sh`，默认：

```bash
DISAGGREGATION_TRANSFER_BACKEND=metadata_only
```

如果 docker 内看不到 CUDA，按仓库说明重启对应 CUDA 12.9 docker 后重试；CUDA/SGLang 环境测试不要在沙箱里跑，需要提权执行 docker 命令。

### 0.13 调试日志

`metadata_only` backend 有独立的日志级别环境变量：

```bash
export SGLANG_METADATA_ONLY_KV_LOG_LEVEL=debug
```

支持 Python logging 的常见级别名或数字级别，例如：

```text
debug
info
warning
error
10
20
```

设置为 `debug` 后，会在这些路径输出传输摘要：

- Decode 发送 metadata：room、prefill rank、decode rank、dst aux index、dst page 数量和前几个 page id、state index 数量、decode prefix len、dummy rank 标记。
- Prefill 接收 metadata：room、decode endpoint、dst aux index、dst page 摘要、state index 摘要、收齐的 destination 数量。
- Prefill 跳过 KV payload：room、chunk page 摘要、chunk slice、是否最后一个 chunk。
- Prefill 发送 aux metadata：room、src/dst aux index、decode endpoint、frame 数量、总字节数、每个 metadata frame 字节数。
- Decode 接收并写入 aux metadata：room、dst aux index、PP rank、frame 摘要，以及写入后状态。

日志不会 dump 完整 KV page 列表或 metadata tensor bytes，只输出摘要，避免长上下文请求时日志过大。

## 1. 最适合你的整体架构

第一版先限制：

```text
P TP == D TP
CP = 1
PP = 1
dense attention
不开 speculative decoding
不开 Mamba/SWA
不开 decode radix cache
```

这样实现会非常干净。

请求生命周期：

```text
                    HTTP bootstrap server
                           ▲
                           │ register rank_ip/rank_port
                           │
               ┌──────── Prefill ─────────┐
               │                          │
               │       real prefill       │
               │                          │
Decode         │                          │
  │            │                          │
  │  META      │                          │
  ├───────────►│                          │
  │ room       │                          │
  │ dst pages  │                          │
  │ aux slot   │                          │
  │ callback   │                          │
  │            │                          │
  │            │   KV transfer = NO-OP    │
  │            │                          │
  │ AUX_DONE   │                          │
  │◄───────────┤                          │
  │ output_id  │                          │
  │ cached_tok │                          │
  │ room       │                          │
  ▼            └──────────────────────────┘
Decode running
```

这样你保留下来的是真实：

- P queue / batching / prefill compute
- D KV preallocation
- bootstrap
- P/D routing
- Decode→Prefill metadata
- Prefill→Decode first-token metadata
- P/D scheduler state machine

去掉的只有：

- KV RDMA/NIXL copy
- state KV copy

这正适合做 **zero-KV-data-plane baseline**。

---

# 2. 不要继承 `FakeKVManager`，要继承 `CommonKVManager`

这是最关键的一点。

现在 `FakeKVManager`：

```python
class FakeKVManager(BaseKVManager):
    ...
    def register_to_bootstrap(self):
        pass
```

它没有真实连接。

而真正 PD 的公共 control plane 已经在：

```python
CommonKVManager
CommonKVSender
CommonKVReceiver
CommonKVBootstrapServer
```

里面实现了。

`CommonKVManager` 会：

- 建立 ZMQ `PULL`
- 获得 `rank_ip`
- 获得 `rank_port`
- Prefill 向 bootstrap server 注册
- Decode 查询 Prefill topology
- 维护 DP/TP/PP rank mapping
- heartbeat
- failure handling

Prefill 注册的信息里已经包括：

```python
payload = {
    "attn_tp_size": ...,
    "attn_tp_rank": ...,
    "rank_ip": self.local_ip,
    "rank_port": self.rank_port,
    "page_size": ...,
    "kv_cache_dtype": ...,
    ...
}
```



所以新的 backend 应该是：

```python
class MetadataOnlyKVManager(CommonKVManager):
    ...

class MetadataOnlyKVSender(CommonKVSender):
    ...

class MetadataOnlyKVReceiver(CommonKVReceiver):
    ...

class MetadataOnlyKVBootstrapServer(CommonKVBootstrapServer):
    pass
```

这样你几乎可以免费复用 SGLang 现在整个 PD control plane。

---

# 3. 新增一个 backend

在：

```text
python/sglang/srt/disaggregation/utils.py
```

加入：

```python
class TransferBackend(Enum):
    MOONCAKE = "mooncake"
    MORI = "mori"
    NIXL = "nixl"
    ASCEND = "ascend"
    FAKE = "fake"

    METADATA_ONLY = "metadata_only"
```

然后在 `get_kv_class()`：

```python
elif transfer_backend == TransferBackend.METADATA_ONLY:
    from sglang.srt.disaggregation.metadata_only import (
        MetadataOnlyKVBootstrapServer,
        MetadataOnlyKVManager,
        MetadataOnlyKVReceiver,
        MetadataOnlyKVSender,
    )

    class_mapping = {
        KVClassType.MANAGER: MetadataOnlyKVManager,
        KVClassType.SENDER: MetadataOnlyKVSender,
        KVClassType.RECEIVER: MetadataOnlyKVReceiver,
        KVClassType.BOOTSTRAP_SERVER: MetadataOnlyKVBootstrapServer,
    }
```

现在 `fake` 故意没有 `BOOTSTRAP_SERVER`。

而你的 backend **必须提供**：

```python
KVClassType.BOOTSTRAP_SERVER
```

因为 Prefill 启动时 `start_disagg_service()` 无条件做：

```python
kv_bootstrap_server_class = get_kv_class(
    transfer_backend,
    KVClassType.BOOTSTRAP_SERVER,
)

bootstrap_server = kv_bootstrap_server_class(...)
```



因此：

```python
class MetadataOnlyKVBootstrapServer(CommonKVBootstrapServer):
    pass
```

就够了。

---

# 4. Decode→Prefill：继续真的发送 metadata

这里可以几乎照抄 `NixlKVReceiver.send_metadata()`。

现在 NIXL Decode 会发送：

```python
sock.send_multipart(
    [
        GUARD,
        str(self.bootstrap_room).encode(),
        self.kv_mgr.local_ip.encode(),
        str(self.kv_mgr.rank_port).encode(),
        self.kv_mgr.agent.name.encode(),
        kv_indices.tobytes(),
        str(aux_index).encode(),
        str(self.required_dst_info_num).encode(),
        packed_state_indices,
        str(decode_prefix_len or 0).encode(),
        ...
    ]
)
```



这个路径非常适合保留。

你可以改成自己的简单 wire format：

```python
sock.send_multipart(
    [
        b"META_ONLY",
        str(self.bootstrap_room).encode(),
        self.kv_mgr.local_ip.encode(),
        str(self.kv_mgr.rank_port).encode(),
        kv_indices.tobytes(),
        str(aux_index).encode(),
        str(self.required_dst_info_num).encode(),
        str(decode_prefix_len or 0).encode(),
    ]
)
```

这里的：

```text
kv_indices
```

我建议**照样发送**。

虽然不会用这些 page indices 做 KV copy，但它能让你的实验保留真实：

```text
Decode preallocation
        ↓
page table construction
        ↓
metadata serialization
        ↓
network metadata transfer
        ↓
Prefill metadata processing
```

这样只是去掉大规模 KV payload。

---

# 5. Decode 的 KV allocation 不需要改

这点很方便。

现在 Decode 在 `send_metadata()` 之前就已经：

```text
allocate KV
↓
req_to_token
↓
kv_indices
↓
page_indices
```

正常执行。

所以例如 64K context：

```text
64K input
 ↓
Decode 真的占用 ~64K token KV capacity
 ↓
但是这些 pages 没有被 Prefill 写入
```

你的 KV pressure 实验仍然有效。

这正是你想要的。

---

# 6. Prefill manager 收到 metadata 后，只建立 room

写一个类似 NIXL `_start_bootstrap_thread()` 的极简版本。

例如：

```python
@dataclass
class MetadataOnlyTransferInfo:
    room: int
    decode_ip: str
    decode_port: int
    dst_aux_index: int
    dst_kv_indices: np.ndarray
    required_dst_info_num: int
    decode_prefix_len: int
```

Prefill manager：

```python
class MetadataOnlyKVManager(CommonKVManager):

    def __init__(self, args, disaggregation_mode, server_args, is_mla_backend=False):
        super().__init__(
            args,
            disaggregation_mode,
            server_args,
            is_mla_backend,
        )

        if disaggregation_mode == DisaggregationMode.PREFILL:
            self.transfer_infos = {}
            self.req_to_decode_prefix_len = {}
            self._start_prefill_listener()

        else:
            self._start_decode_listener()
            self._start_heartbeat_checker_thread()
```

Prefill listener：

```python
def _start_prefill_listener(self):

    def loop():
        while True:
            msg = self.server_socket.recv_multipart()

            if msg[0] != b"META_ONLY":
                continue

            room = int(msg[1])
            decode_ip = msg[2].decode()
            decode_port = int(msg[3])
            dst_kv_indices = np.frombuffer(
                msg[4],
                dtype=np.int32,
            )
            dst_aux_index = int(msg[5])
            required = int(msg[6])
            decode_prefix_len = int(msg[7])

            info = MetadataOnlyTransferInfo(...)

            self.transfer_infos.setdefault(room, [])
            self.transfer_infos[room].append(info)

            if len(self.transfer_infos[room]) == required:
                self.req_to_decode_prefix_len[room] = decode_prefix_len

                self.update_status(
                    room,
                    KVPoll.WaitingForInput,
                )

    threading.Thread(target=loop, daemon=True).start()
```

这就对应了真实 NIXL 的：

```text
Decode send metadata
        ↓
Prefill receives TransferInfo
        ↓
room bootstrapped
        ↓
Prefill sender.poll() = WaitingForInput
        ↓
开始真正 Prefill
```

NIXL 当前也是在收齐目标端 info 后把 room 设成 `WaitingForInput`。

---

# 7. Sender：最大的区别就是 `send()` 什么都不 copy

你的 Sender 可以直接继承：

```python
CommonKVSender
```

因为它已经有：

```python
_prepare_send_indices()
pop_decode_prefix_len()
init()
timeout handling
clear()
```



实现：

```python
class MetadataOnlyKVSender(CommonKVSender):

    def __init__(...):
        super().__init__(...)
        self.init_time = time.time()

    def send(
        self,
        kv_indices,
        state_indices=None,
        num_kv_tokens=None,
    ):
        kv_indices, index_slice, is_last_chunk, should_skip = (
            self._prepare_send_indices(
                kv_indices,
                state_indices,
            )
        )

        if should_skip:
            return

        # Important:
        # DO NOT COPY KV.
        self._record_transfer_indices(
            kv_indices,
            state_indices,
        )

        if is_last_chunk:
            self._send_aux_metadata()
```

也就是说：

```text
正常 NIXL:

sender.send()
   ↓
transfer_worker
   ↓
send_kvcache()
   ↓
RDMA WRITE


metadata_only:

sender.send()
   ↓
NO-OP
```

Prefill 本身的 `send_kv_chunk()` 完全可以不改。

它仍然正常算：

```python
page_indices = kv_to_page_indices(...)

req.disagg_kv_sender.send(
    page_indices,
    ...
)
```

只是新的 sender 把它吃掉。

这是一层非常干净的 abstraction boundary。

---

# 8. 但是最后一个 chunk 必须真的传 Prefill metadata

这是实现里最容易漏的地方。

Prefill 在最后一个 chunk 前会：

```python
self.disagg_metadata_buffers.set_buf(req)
```



里面包含：

```text
output_ids
cached_tokens
cached_tokens_device
cached_tokens_host
cached_tokens_storage

output token logprob
top logprobs

sampling mask

spec topk
spec hidden state

bootstrap_room
```



所以你**不要 fake 掉这个 metadata**。

---

# 9. 第一版只传最小 metadata 就够了

如果你的实验满足：

```text
不需要 correctness
不开 return_logprob
不开 speculative decoding
普通 dense model
```

那么最小只需要：

```text
output_id
cached_tokens[0:7]
bootstrap_room
```

例如 Prefill：

```python
def _send_aux_metadata(self):

    src_idx = self.aux_index

    mbuf = self.kv_mgr.metadata_buffers

    output_id = int(
        mbuf.output_ids[src_idx, 0].item()
    )

    cached_tokens = (
        mbuf.cached_tokens[src_idx, :7]
        .cpu()
        .numpy()
        .astype(np.int32)
    )

    for info in self.kv_mgr.transfer_infos[self.bootstrap_room]:

        sock = ...

        sock.send_multipart(
            [
                b"AUX_DONE",
                str(self.bootstrap_room).encode(),
                str(info.dst_aux_index).encode(),
                str(output_id).encode(),
                cached_tokens.tobytes(),
            ]
        )

    self.kv_mgr.update_status(
        self.bootstrap_room,
        KVPoll.Success,
    )
```

注意顺序：

```text
send AUX
   ↓
then
Prefill Success
```

不要反过来。

否则 Prefill 可能先释放：

```text
metadata_buffer_index
```

导致 race。

---

# 10. 需要让 Manager 拿到 `MetadataBuffers`

目前 `KVArgs` 只有这些 buffer 的 pointer：

```python
kv_args.aux_data_ptrs,
kv_args.aux_data_lens,
kv_args.aux_item_lens
    = metadata_buffers.get_buf_infos()
```

Prefill 和 Decode 都这么做。

对于 RDMA backend，pointer 足够。

但你的 Python/ZMQ backend 想直接读取 Tensor 内容，所以建议加：

```python
def set_metadata_buffers(self, metadata_buffers):
    self.metadata_buffers = metadata_buffers
```

然后 Prefill `_init_kv_manager()`：

```python
kv_manager = kv_manager_class(...)

if hasattr(kv_manager, "set_metadata_buffers"):
    kv_manager.set_metadata_buffers(
        self.metadata_buffers
    )
```

Decode 也一样。

这是我认为你需要改现有 SGLang core 的**唯一一个稍微不漂亮但非常实用的点**。

---

# 11. Decode 收到 AUX 后，写入本地 metadata slot

Decode manager 起一个真实 listener：

```python
def _start_decode_listener(self):

    def loop():

        while True:
            msg = self.server_socket.recv_multipart()

            if msg[0] != b"AUX_DONE":
                continue

            room = int(msg[1])
            aux_idx = int(msg[2])
            output_id = int(msg[3])
            cached_tokens = np.frombuffer(
                msg[4],
                dtype=np.int32,
            )

            mbuf = self.metadata_buffers

            mbuf.output_ids[
                aux_idx, 0
            ] = output_id

            mbuf.cached_tokens[
                aux_idx, :7
            ].copy_(
                torch.from_numpy(
                    cached_tokens.copy()
                )
            )

            # Very important
            mbuf.bootstrap_room[
                aux_idx, 0
            ] = room

            # Metadata must be written BEFORE success
            self.update_status(
                room,
                KVPoll.Success,
            )

    threading.Thread(
        target=loop,
        daemon=True,
    ).start()
```

这里：

```python
bootstrap_room
```

尤其重要。

---

# 12. 不要把你的 backend 加进 `_is_fake_transfer()`

这一点非常关键。

当前：

```python
def _is_fake_transfer(req):
    return req.bootstrap_host == FAKE_BOOTSTRAP_HOST or (
        req.bootstrap_host is None
        and backend == "fake"
    )
```



而 metadata readiness gate：

```python
if poll == Success:
    if _is_fake_transfer(req):
        continue

    actual_room =
        metadata_buffers.bootstrap_room[idx, 0]

    if actual_room == 0:
        polls[i] = Transferring
```

也就是说真实 backend 必须满足：

```text
metadata really arrived
        ↓
bootstrap_room written
        ↓
才允许 Success
```

这非常适合你的实验。

所以：

```python
metadata_only
```

**不要被 `_is_fake_transfer()` 判断为 True。**

否则你刚建立的真实 metadata synchronization 又被绕开了。

---

# 13. Receiver 可以非常简单

```python
class MetadataOnlyKVReceiver(CommonKVReceiver):

    def __init__(...):
        super().__init__(...)
        self.started_transfer = False
        self.init_time = None

    def _register_kv_args(self):
        # No memory registration.
        return True
```

这里正好利用 `CommonKVReceiver`。

它已经会：

```text
query bootstrap server
↓
resolve prefill DP/TP rank
↓
get rank_ip/rank_port
↓
建立 connection_pool
```



然后 `send_metadata()` 就按前面说的发：

```text
META_ONLY
```

最后：

```python
def poll(self):

    if self.conclude_state is not None:
        return self.conclude_state

    status = self.kv_mgr.check_status(
        self.bootstrap_room
    )

    if status in (
        KVPoll.Success,
        KVPoll.Failed,
    ):
        self.conclude_state = status
        return status

    if self.started_transfer:
        timeout = self._check_waiting_timeout()
        if timeout is not None:
            return timeout

    return status
```

不需要 NIXL 的：

```text
get_new_notifs()
transfer_statuses
RDMA completion
```

---

# 14. 这样之后 Decode 的执行路径完全不用改

Decode 当前 lifecycle 本来就是：

```text
PreallocQueue
 ↓
allocate KV
 ↓
receiver.send_metadata()
 ↓
TransferQueue
 ↓
receiver.poll()
 ↓
Success
 ↓
PrebuiltExtendBatch
 ↓
跳过 Prefill forward
 ↓
RunningBatch
```



你的 backend 只是把：

```text
TransferQueue 等待条件
```

从：

```text
KV RDMA completed
+
aux metadata completed
```

变成：

```text
aux metadata completed
```

所以非常符合你的实验需求。

---

# 15. 最终你真正测到的时间

这时真实运行时间就是：

\[
T =
T_{Pqueue}
+
T_{Prefill}
+
T_{D\ prealloc}
+
T_{metadata}
+
T_{Decode}
\]

而没有：

\[
T_{KVTransfer}
\]

这比当前 decode-only fake 好很多。

当前 fake 测的是：

```text
D prealloc
+
D scheduling
+
Decode compute
```

你的 `metadata_only` 则可以测：

```text
真实 P scheduling
+
真实 Prefill compute
+
真实 P/D synchronization
+
真实 D preallocation
+
真实 metadata control-plane
+
真实 Decode scheduling/compute
-
KV data-plane
```

因此：

```text
metadata_only
       vs
NIXL
```

的差：

\[
T_{\text{NIXL}}-T_{\text{metadata-only}}
\]

会是一个相当干净的 **KV data-plane transfer + associated contention/overlap cost** 估计。

---

## 我建议的第一版文件改动

| 文件 | 改动 |
|---|---|
| `disaggregation/utils.py` | 加 `METADATA_ONLY` + class mapping |
| `disaggregation/metadata_only/__init__.py` | export 4 个 class |
| `disaggregation/metadata_only/conn.py` | Manager/Sender/Receiver/BootstrapServer |
| `prefill.py` | manager 创建后塞入 `metadata_buffers` |
| `decode.py` | manager 创建后塞入 `metadata_buffers` |
| `server_args.py` | 如果 backend choices 是静态列表，加 `metadata_only` |
| tests | metadata wire + PD 2-GPU smoke test |

**不需要改 `send_kv_chunk()`，也不需要改 Decode preallocation。**

我尤其建议第一版**不要支持 state cache / speculative / heterogeneous TP**。先把 `Qwen3/Llama + TP_P=TP_D + PP1 + CP1` 跑通；这大概是最少侵入 SGLang、同时实验语义最干净的实现路线。
