from __future__ import annotations

from array import array

import torch
import pytest

from sglang.srt.layers.attention.flashinfer_cascade_backend import (
    FlashInferCascadeAttnBackend,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams


def test_req_parses_cascade_custom_params():
    sampling_params = SamplingParams(
        max_new_tokens=1,
        custom_params={
            "cascade": {
                "prefix_ref_rid": "anchor-rid",
                "shared_prefix_len": 12,
                "system_prefix_len": 4,
            }
        },
    )
    req = Req(
        rid="child-rid",
        origin_input_text="",
        origin_input_ids=array("q", [1, 2, 3]),
        sampling_params=sampling_params,
    )

    assert req.cascade_prefix_ref_rid == "anchor-rid"
    assert req.cascade_shared_prefix_len == 12
    assert req.cascade_system_prefix_len == 4


def test_three_level_metadata_groups_non_contiguous_requests():
    backend = FlashInferCascadeAttnBackend.__new__(FlashInferCascadeAttnBackend)
    req_pool_indices = torch.arange(4)
    seq_lens_cpu = torch.tensor([16, 16, 16, 16], dtype=torch.int32)

    meta = backend._build_three_level_metadata(
        bs=4,
        req_pool_indices=req_pool_indices,
        seq_lens_cpu=seq_lens_cpu,
        rids=["a", "b", "c", "d"],
        prefix_ref_rids=[None, "a", None, "a"],
        shared_prefix_lens=[None, 12, None, 12],
        system_prefix_lens=[4, 4, 4, 4],
        fallback_common_prefix=4,
    )

    assert meta is not None
    assert meta["system_prefix"] == 4
    assert meta["perm"] == [0, 1, 3, 2]
    assert meta["inv_perm"] == [0, 1, 3, 2]
    assert meta["q_indptr_l1_cpu"] == [0, 1, 3, 4]
    assert [g["members"] for g in meta["groups"]] == [[0], [1, 3], [2]]
    assert [g["shared_prefix"] for g in meta["groups"]] == [4, 12, 4]


def test_three_level_metadata_clamps_invalid_lengths_and_short_lists():
    backend = FlashInferCascadeAttnBackend.__new__(FlashInferCascadeAttnBackend)
    meta = backend._build_three_level_metadata(
        bs=3,
        req_pool_indices=torch.arange(3),
        seq_lens_cpu=torch.tensor([16, 10, 8], dtype=torch.int32),
        rids=["a"],
        prefix_ref_rids=[None, "a"],
        shared_prefix_lens=["bad", 32],
        system_prefix_lens=[4],
        fallback_common_prefix=4,
    )

    assert meta is not None
    assert meta["system_prefix"] == 4
    # request 1 points to request 0, but its own seq len clamps shared len to 9.
    assert meta["per_req_shared"] == [4, 9, 4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flashinfer_three_level_cuda_graph_wrapper_smoke():
    from flashinfer import MultiLevelCascadeAttentionWrapper

    bs, num_heads, num_kv_heads, head_dim, pages = 4, 4, 4, 128, 64
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    q = torch.randn(bs, num_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(
        pages, 1, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
    )
    v = torch.randn(
        pages, 1, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
    )

    qo0 = torch.tensor([0, bs], dtype=torch.int32, device="cuda")
    ki0 = torch.zeros(2, dtype=torch.int32, device="cuda")
    idx0 = torch.zeros(16, dtype=torch.int32, device="cuda")
    lp0 = torch.ones(1, dtype=torch.int32, device="cuda")
    qo1 = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
    ki1 = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
    idx1 = torch.zeros(bs * 16, dtype=torch.int32, device="cuda")
    lp1 = torch.zeros(bs, dtype=torch.int32, device="cuda")
    qo2 = torch.arange(bs + 1, dtype=torch.int32, device="cuda")
    ki2 = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
    idx2 = torch.zeros(bs * 16, dtype=torch.int32, device="cuda")
    lp2 = torch.ones(bs, dtype=torch.int32, device="cuda")

    wrapper = MultiLevelCascadeAttentionWrapper(
        3,
        workspace,
        "NHD",
        use_cuda_graph=True,
        qo_indptr_buf_arr=[qo0, qo1, qo2],
        paged_kv_indptr_buf_arr=[ki0, ki1, ki2],
        paged_kv_indices_buf_arr=[idx0, idx1, idx2],
        paged_kv_last_page_len_buf_arr=[lp0, lp1, lp2],
    )

    def fill_plan(level1_start: int) -> None:
        ki0.copy_(torch.tensor([0, 4], dtype=torch.int32, device="cuda"))
        idx0[:4].copy_(torch.arange(4, dtype=torch.int32, device="cuda"))
        lp0.fill_(1)

        qo1.copy_(torch.tensor([0, 1, 3, 4, 4], dtype=torch.int32, device="cuda"))
        ki1.copy_(torch.tensor([0, 0, 4, 4, 4], dtype=torch.int32, device="cuda"))
        idx1[:4].copy_(
            torch.arange(
                level1_start,
                level1_start + 4,
                dtype=torch.int32,
                device="cuda",
            )
        )
        lp1.copy_(torch.tensor([0, 1, 0, 0], dtype=torch.int32, device="cuda"))

        ki2.copy_(torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32, device="cuda"))
        idx2[:16].copy_(torch.arange(8, 24, dtype=torch.int32, device="cuda"))
        lp2.fill_(1)
        wrapper.plan(
            [qo0, qo1, qo2],
            [ki0, ki1, ki2],
            [idx0[:4], idx1[:4], idx2[:16]],
            [lp0, lp1, lp2],
            num_heads,
            num_kv_heads,
            head_dim,
            1,
            False,
            "NONE",
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
        )

    fill_plan(4)
    for _ in range(2):
        wrapper.run(q, (k, v))
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = wrapper.run(q, (k, v))
    fill_plan(5)
    graph.replay()
    torch.cuda.synchronize()

    assert tuple(out.shape) == (bs, num_heads, head_dim)
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_backend_three_level_cg_fill_and_run_smoke():
    bs, num_heads, num_kv_heads, head_dim, pages = 4, 4, 4, 128, 64
    backend = FlashInferCascadeAttnBackend.__new__(FlashInferCascadeAttnBackend)
    backend._device = torch.device("cuda")
    backend.workspace_buffer = torch.empty(
        256 * 1024 * 1024, dtype=torch.uint8, device="cuda"
    )
    backend._cg_cascade_wrappers = {}
    backend._cg_cascade_buffers = {}
    backend._cg_max_shared_pages = 16
    backend._cg_max_pages_per_req = 16
    backend.req_to_token = (
        torch.arange(pages, dtype=torch.int64, device="cuda").view(1, -1).repeat(bs, 1)
    )
    backend.num_qo_heads = num_heads
    backend.num_kv_heads_local = num_kv_heads
    backend.head_dim_local = head_dim
    backend.cascade_page_size = 1
    backend.q_dtype = torch.float16
    backend.kv_dtype = torch.float16
    backend._dbg_enabled = False

    backend._allocate_cg_cascade_for_bs(bs)
    ok = backend._fill_cg_cascade_plan(
        bs=bs,
        req_pool_indices=torch.arange(bs, dtype=torch.int64, device="cuda"),
        seq_lens_cpu=torch.tensor([16, 16, 16, 16], dtype=torch.int32),
        common_prefix_tokens=4,
        rids=["a", "b", "c", "d"],
        prefix_ref_rids=[None, "a", None, "a"],
        shared_prefix_lens=[None, 12, None, 12],
        system_prefix_lens=[4, 4, 4, 4],
    )

    assert ok
    bufs = backend._cg_cascade_buffers[bs]
    assert bufs["perm"].tolist() == [0, 1, 3, 2]
    assert bufs["qo_indptr_l1"].tolist() == [0, 1, 3, 4, 4]

    q = torch.randn(bs, num_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(
        pages, 1, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
    )
    v = torch.randn(
        pages, 1, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
    )
    q_reordered = q.index_select(0, bufs["perm"])
    out = backend._cg_cascade_wrappers[bs].run(q_reordered, (k, v))
    out = out.index_select(0, bufs["inv_perm"])
    torch.cuda.synchronize()

    assert tuple(out.shape) == (bs, num_heads, head_dim)
    assert torch.isfinite(out).all()
