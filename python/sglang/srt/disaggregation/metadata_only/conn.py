from __future__ import annotations

import dataclasses
import logging
import os
import threading
import time
import uuid
from typing import List, Optional

import numpy as np
import numpy.typing as npt
import torch
import zmq

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
    KVTransferError,
)
from sglang.srt.disaggregation.common.utils import pack_int_lists, unpack_int_lists
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import NetworkAddress

logger = logging.getLogger(__name__)

GUARD = "NixlMsgGuard".encode("ascii")
AUX_GUARD = "MetadataOnlyAux".encode("ascii")
LOG_LEVEL_ENV = "SGLANG_METADATA_ONLY_KV_LOG_LEVEL"


def _apply_log_level_from_env() -> None:
    raw_level = os.getenv(LOG_LEVEL_ENV)
    if not raw_level:
        return

    level_name = raw_level.strip().upper()
    if level_name.isdigit():
        level = int(level_name)
    else:
        level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        logger.warning(
            "Invalid %s=%r for metadata-only KV transfer logging",
            LOG_LEVEL_ENV,
            raw_level,
        )
        return
    logger.setLevel(level)


def _array_summary(arr: npt.NDArray[np.int32], max_items: int = 8) -> str:
    if arr is None:
        return "none"
    preview = arr[:max_items].tolist()
    suffix = "..." if len(arr) > max_items else ""
    return f"len={len(arr)} preview={preview}{suffix}"


def _state_indices_summary(state_indices: Optional[List]) -> str:
    if not state_indices:
        return "none"
    lengths = [len(indices) if indices is not None else 0 for indices in state_indices]
    return f"components={len(state_indices)} lengths={lengths}"


def _frame_summary(frames: List[bytes]) -> str:
    sizes = [len(frame) for frame in frames]
    return f"count={len(frames)} total_bytes={sum(sizes)} sizes={sizes}"


_apply_log_level_from_env()


@dataclasses.dataclass
class MetadataOnlyTransferInfo:
    room: int
    endpoint: str
    dst_port: int
    agent_name: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    required_dst_info_num: int
    dst_state_indices: List[List[int]]
    decode_prefix_len: Optional[int] = None
    is_dummy_rank: Optional[bool] = None

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        dst_state_indices = (
            unpack_int_lists(msg[7], "i") if len(msg) > 7 and msg[7] != b"" else []
        )
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            agent_name=msg[3].decode("ascii"),
            dst_kv_indices=np.frombuffer(msg[4], dtype=np.int32),
            dst_aux_index=int(msg[5].decode("ascii")),
            required_dst_info_num=int(msg[6].decode("ascii")),
            dst_state_indices=dst_state_indices,
            decode_prefix_len=(
                int(msg[8].decode("ascii")) if len(msg) > 8 and msg[8] != b"" else None
            ),
            is_dummy_rank=(
                bool(int(msg[9].decode("ascii")))
                if len(msg) > 9 and msg[9] != b""
                else None
            ),
        )


def _metadata_slots(metadata_buffers, idx: int):
    return (
        metadata_buffers.output_ids[idx],
        metadata_buffers.cached_tokens[idx],
        metadata_buffers.output_token_logprobs_val[idx],
        metadata_buffers.output_token_logprobs_idx[idx],
        metadata_buffers.output_top_logprobs_val[idx],
        metadata_buffers.output_top_logprobs_idx[idx],
        (
            metadata_buffers.output_token_sampling_mask_len[idx]
            if metadata_buffers.enable_sampling_mask
            else None
        ),
        (
            metadata_buffers.output_token_sampling_mask_idx[idx]
            if metadata_buffers.enable_sampling_mask
            else None
        ),
        (
            metadata_buffers.output_token_sampling_logprobs[idx]
            if metadata_buffers.enable_sampling_mask
            else None
        ),
        metadata_buffers.output_topk_p[idx],
        metadata_buffers.output_topk_index[idx],
        metadata_buffers.output_hidden_states[idx],
        (
            metadata_buffers.output_dsa_topk_indices[idx]
            if metadata_buffers.output_dsa_topk_indices is not None
            else None
        ),
        metadata_buffers.bootstrap_room[idx],
    )


def _tensor_to_bytes(tensor: Optional[torch.Tensor]) -> bytes:
    if tensor is None:
        return b""
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def _copy_bytes_to_tensor(frame: bytes, dst: Optional[torch.Tensor]) -> None:
    if dst is None:
        return
    if len(frame) != dst.nbytes:
        raise ValueError(
            f"Metadata frame size mismatch: got {len(frame)} bytes, "
            f"expected {dst.nbytes}"
        )
    byte_tensor = torch.frombuffer(bytearray(frame), dtype=torch.uint8)
    src = byte_tensor.view(dst.dtype).view(dst.shape)
    dst.copy_(src.to(device=dst.device))


class MetadataOnlyKVManager(CommonKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        _apply_log_level_from_env()
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        # Match scheduler batch logging: only attention TP rank 0 emits routine
        # logs. Errors still use logger.error/exception on the rank that sees them.
        self.is_stats_logging_rank = self.attn_tp_rank == 0
        self.agent_name = f"metadata_only_{uuid.uuid4()}"
        self.metadata_buffers = None
        self.enable_staging = False
        self.log_debug(
            "metadata_only manager initialized: mode=%s agent_name=%s "
            "rank_ip=%s rank_port=%s tp_rank=%s cp_rank=%s pp_rank=%s",
            self.disaggregation_mode,
            self.agent_name,
            self.local_ip,
            self.rank_port,
            self.attn_tp_rank,
            self.attn_cp_rank,
            self.pp_rank,
        )

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._start_bootstrap_thread()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self._start_decode_thread()
            self._start_heartbeat_checker_thread()
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

    def log_debug(self, msg: str, *args) -> None:
        if self.is_stats_logging_rank:
            logger.debug(msg, *args)

    def set_metadata_buffers(self, metadata_buffers) -> None:
        self.metadata_buffers = metadata_buffers
        self.log_debug(
            "metadata_only metadata buffers registered: mode=%s slots=%s "
            "sampling_mask=%s dsa_topk=%s",
            self.disaggregation_mode,
            metadata_buffers.output_ids.shape[0],
            metadata_buffers.enable_sampling_mask,
            metadata_buffers.output_dsa_topk_indices is not None,
        )

    def _handle_abort_notification(self, msg: List[bytes]) -> bool:
        if not msg or msg[0] != b"ABORT":
            return False
        try:
            room = int(msg[1].decode("ascii"))
        except Exception:
            self.log_debug("Ignoring malformed metadata-only abort notification")
            return True
        if room in self.request_status and self.check_status(room) != KVPoll.Success:
            self.record_failure(room, "Aborted by peer notification.")
            self.update_status(room, KVPoll.Failed)
        return True

    def _start_bootstrap_thread(self) -> None:
        def bootstrap_thread():
            while True:
                msg = self.server_socket.recv_multipart()
                if self._handle_abort_notification(msg):
                    continue
                if not msg or msg[0] != GUARD:
                    self.log_debug("Ignoring non metadata-only bootstrap message")
                    continue

                waiting_req_bytes = msg[1:]
                room_str = waiting_req_bytes[0].decode("ascii")
                if room_str == "None":
                    # metadata_only does not need decode-side pointer registration.
                    continue

                room = int(room_str)
                agent_name = waiting_req_bytes[3].decode("ascii")
                if room not in self.transfer_infos:
                    self.transfer_infos[room] = {}
                self.transfer_infos[room][agent_name] = (
                    MetadataOnlyTransferInfo.from_zmq(waiting_req_bytes)
                )
                info = self.transfer_infos[room][agent_name]
                required_dst_info_num = self.transfer_infos[room][
                    agent_name
                ].required_dst_info_num
                self.log_debug(
                    "metadata_only received decode metadata: room=%s "
                    "agent=%s decode_endpoint=%s:%s dst_aux_index=%s "
                    "dst_pages=%s state_indices=%s decode_prefix_len=%s "
                    "is_dummy=%s required_dst_info_num=%s received=%s",
                    room,
                    agent_name,
                    info.endpoint,
                    info.dst_port,
                    info.dst_aux_index,
                    _array_summary(info.dst_kv_indices),
                    _state_indices_summary(info.dst_state_indices),
                    info.decode_prefix_len,
                    info.is_dummy_rank,
                    required_dst_info_num,
                    len(self.transfer_infos[room]),
                )
                if len(self.transfer_infos[room]) == required_dst_info_num:
                    self.req_to_decode_prefix_len[room] = next(
                        (
                            info.decode_prefix_len
                            for info in self.transfer_infos[room].values()
                            if info.decode_prefix_len is not None
                        ),
                        0,
                    )
                    self.log_debug("metadata_only room=%s is bootstrapped", room)
                    self.update_status(room, KVPoll.WaitingForInput)

        threading.Thread(target=bootstrap_thread, daemon=True).start()

    def _start_decode_thread(self) -> None:
        def decode_thread():
            while True:
                msg = self.server_socket.recv_multipart()
                if self._handle_abort_notification(msg):
                    continue
                if not msg or msg[0] != AUX_GUARD:
                    self.log_debug("Ignoring non metadata-only aux message")
                    continue

                try:
                    room = int(msg[1].decode("ascii"))
                    dst_aux_index = int(msg[2].decode("ascii"))
                    pp_rank = int(msg[3].decode("ascii"))
                    frames = msg[4:]
                    self.log_debug(
                        "metadata_only received aux metadata: room=%s "
                        "dst_aux_index=%s pp_rank=%s frames=%s",
                        room,
                        dst_aux_index,
                        pp_rank,
                        _frame_summary(frames),
                    )
                    if self.metadata_buffers is None:
                        raise RuntimeError("metadata buffers are not registered")
                    slots = _metadata_slots(self.metadata_buffers, dst_aux_index)
                    if len(frames) != len(slots):
                        raise ValueError(
                            f"Metadata frame count mismatch: got {len(frames)}, "
                            f"expected {len(slots)}"
                        )
                    for frame, slot in zip(frames, slots):
                        _copy_bytes_to_tensor(frame, slot)
                    self.metadata_buffers.bootstrap_room[dst_aux_index, 0] = room
                    self.update_status(room, KVPoll.Success)
                    self.log_debug(
                        "metadata_only applied aux metadata: room=%s "
                        "dst_aux_index=%s status=%s",
                        room,
                        dst_aux_index,
                        self.check_status(room),
                    )
                except Exception as exc:
                    logger.exception("Failed to apply metadata-only aux payload")
                    if "room" in locals():
                        self.record_failure(room, str(exc))
                        self.update_status(room, KVPoll.Failed)

        threading.Thread(target=decode_thread, daemon=True).start()


class MetadataOnlyKVSender(CommonKVSender):
    def __init__(
        self,
        mgr: MetadataOnlyKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
        req_has_disagg_prefill_dp_rank: bool = False,
    ):
        super().__init__(
            mgr,
            bootstrap_addr,
            bootstrap_room,
            dest_tp_ranks,
            pp_rank,
            req_has_disagg_prefill_dp_rank,
        )
        self._transfer_start_time: Optional[float] = None
        self.init_time = time.time()

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
        num_kv_tokens: Optional[int] = None,
    ):
        kv_indices, index_slice, is_last_chunk, should_skip = (
            self._prepare_send_indices(kv_indices, state_indices)
        )
        if should_skip:
            return
        if self._transfer_start_time is None:
            self._transfer_start_time = time.perf_counter()
        self._record_transfer_indices(kv_indices, state_indices)
        self.kv_mgr.log_debug(
            "metadata_only skipped KV payload transfer: room=%s chunk_pages=%s "
            "state_indices=%s index_slice=(%s,%s) is_last_chunk=%s "
            "num_kv_tokens=%s",
            self.bootstrap_room,
            _array_summary(kv_indices),
            _state_indices_summary(state_indices),
            index_slice.start,
            index_slice.stop,
            is_last_chunk,
            num_kv_tokens,
        )
        if is_last_chunk:
            self._send_aux_metadata()
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Success)

    def _send_aux_metadata(self) -> None:
        if self.aux_index is None:
            raise RuntimeError("Missing aux index for metadata-only final chunk")
        if self.kv_mgr.metadata_buffers is None:
            raise RuntimeError("metadata buffers are not registered")

        room_infos = self.kv_mgr.transfer_infos.get(self.bootstrap_room)
        if not room_infos:
            raise RuntimeError(
                f"Missing metadata-only transfer info for room {self.bootstrap_room}"
            )

        frames = [
            _tensor_to_bytes(slot)
            for slot in _metadata_slots(self.kv_mgr.metadata_buffers, self.aux_index)
        ]
        for info in room_infos.values():
            endpoint = NetworkAddress(info.endpoint, info.dst_port).to_tcp()
            self.kv_mgr.log_debug(
                "metadata_only sending aux metadata: room=%s src_aux_index=%s "
                "dst_aux_index=%s endpoint=%s pp_rank=%s frames=%s",
                self.bootstrap_room,
                self.aux_index,
                info.dst_aux_index,
                endpoint,
                self.kv_mgr.pp_rank,
                _frame_summary(frames),
            )
            self.kv_mgr._send_multipart_locked(
                endpoint,
                [
                    AUX_GUARD,
                    str(self.bootstrap_room).encode("ascii"),
                    str(info.dst_aux_index).encode("ascii"),
                    str(self.kv_mgr.pp_rank).encode("ascii"),
                    *frames,
                ],
            )

    def poll(self) -> KVPoll:
        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status == KVPoll.Bootstrapping:
            timeout_result = self._check_bootstrap_timeout()
            if timeout_result is not None:
                return timeout_result
        if (
            status == KVPoll.Success
            and self._transfer_start_time is not None
            and self._transfer_metric.transfer_latency_s is None
        ):
            self._transfer_metric.transfer_latency_s = (
                time.perf_counter() - self._transfer_start_time
            )
        return status

    def failure_exception(self):
        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(self.bootstrap_room, None)
        self.clear()
        raise KVTransferError(
            self.bootstrap_room,
            failure_reason or "Metadata-only KVSender Exception",
            is_from_another_rank=failure_reason is None,
        )


class MetadataOnlyKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: MetadataOnlyKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        self.started_transfer = False
        super().__init__(mgr, bootstrap_addr, bootstrap_room)

    def _register_kv_args(self) -> bool:
        return True

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
    ):
        if self.bootstrap_infos is None:
            logger.error(
                f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return

        packed_state_indices = (
            pack_int_lists(
                [(idx if idx is not None else []) for idx in state_indices],
                "i",
            )
            if state_indices is not None
            else b""
        )
        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            is_dummy = bootstrap_info["is_dummy"]
            self.kv_mgr.log_debug(
                "metadata_only sending decode metadata: room=%s "
                "prefill_rank=%s:%s decode_rank=%s:%s aux_index=%s "
                "dst_pages=%s state_indices=%s packed_state_bytes=%s "
                "decode_prefix_len=%s is_dummy=%s required_dst_info_num=%s",
                self.bootstrap_room,
                bootstrap_info.get("rank_ip"),
                bootstrap_info.get("rank_port"),
                self.kv_mgr.local_ip,
                self.kv_mgr.rank_port,
                aux_index,
                _array_summary(kv_indices if not is_dummy else np.array([], dtype=np.int32)),
                _state_indices_summary(state_indices if not is_dummy else None),
                len(packed_state_indices) if not is_dummy else 0,
                decode_prefix_len or 0,
                is_dummy,
                self.required_dst_info_num,
            )
            try:
                with lock:
                    sock.send_multipart(
                        [
                            GUARD,
                            str(self.bootstrap_room).encode("ascii"),
                            self.kv_mgr.local_ip.encode("ascii"),
                            str(self.kv_mgr.rank_port).encode("ascii"),
                            self.kv_mgr.agent_name.encode("ascii"),
                            kv_indices.tobytes() if not is_dummy else b"",
                            str(aux_index).encode("ascii"),
                            str(self.required_dst_info_num).encode("ascii"),
                            packed_state_indices if not is_dummy else b"",
                            str(decode_prefix_len or 0).encode("ascii"),
                            str(int(is_dummy)).encode("ascii"),
                        ]
                    )
            except zmq.ZMQError:
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    f"send_metadata to prefill {bootstrap_info.get('rank_ip')}:{bootstrap_info.get('rank_port')} failed",
                )
                self.conclude_state = KVPoll.Failed
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return

        self.started_transfer = True
        self.init_time = time.time()

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
            return status
        if not self.started_transfer:
            return status
        timeout_result = self._check_waiting_timeout()
        if timeout_result is not None:
            return timeout_result
        return KVPoll.WaitingForInput

    def failure_exception(self):
        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(self.bootstrap_room, None)
        raise KVTransferError(
            self.bootstrap_room,
            failure_reason or "Metadata-only KVReceiver Exception",
            is_from_another_rank=failure_reason is None,
        )


class MetadataOnlyKVBootstrapServer(CommonKVBootstrapServer):
    pass
