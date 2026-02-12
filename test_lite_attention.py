import os

import pytest
import torch
from lite_attention import LiteAttention

pytestmark = pytest.mark.filterwarnings("ignore:Module has no registry. Using local config.")

# Enable debug mode to allow non-negative thresholds in tests
os.environ["LITE_ATTENTION_DEBUG"] = "TRUE"

HEAD_DIMS = [32, 64, 96, 128, 192, 256]
BATCH = 2
SEQ_LEN = 18200
HEADS = 32

SKIP_CASES = [
    pytest.param(lambda s: [0, 1000, 10000, s - 1], id="begin_and_end"),
    pytest.param(lambda s: [0, 5000], id="first_half"),
    pytest.param(lambda s: [s // 4, s // 2], id="middle_quarter"),
    pytest.param(lambda s: [0, s // 10, s * 9 // 10, s - 1], id="first_last_10pct"),
    pytest.param(lambda s: [s // 3, s * 2 // 3], id="middle_third"),
    pytest.param(lambda s: [0, 2000, 5000, 7000, 10000, s - 1], id="multi_small"),
]
DO_CASES = [
    *SKIP_CASES,
    pytest.param(lambda s: [0, 2000, 15000, s - 1], id="custom"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_test_tensors(batch, seq_len, heads, head_dim):
    q = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    return q, k, v


def generate_rectangular_test_tensors(batch, q_len, k_len, heads, head_dim):
    q = torch.randn(batch, q_len, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, k_len, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, k_len, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    return q, k, v


def warmup(attn, q, k, v, num_iters=1):
    for _ in range(num_iters):
        torch.cuda.synchronize()
        output = attn(q, k, v)
        torch.cuda.synchronize()
    return output


def assert_skip_list_length_valid(skip_list):
    """List length field must not exceed the skip list dimension."""
    assert (skip_list.shape[-1] > skip_list[..., 0]).all()


def assert_no_empty_or_negative_ranges(skip_list):
    """All ranges in the skip list must be non-empty and positive."""
    diff = skip_list[..., 1:-1] - skip_list[..., 2:]
    sign = torch.sign(diff.flatten()[0])
    diff = (diff * sign) > 0
    arange = torch.arange(diff.shape[-1], device=skip_list.device).view(1, 1, 1, -1) >= skip_list[..., 0:1] - 1
    assert ((arange + diff) > 0).all(-1).all()


def compute_reference_lse(q, k, head_dim):
    scale = 1.0 / (head_dim**0.5)
    q_ref = q.transpose(1, 2).float()
    k_ref = k.transpose(1, 2).float()
    scores = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * scale
    return torch.logsumexp(scores, dim=-1)


def compute_reference_attention_output(q, k, v, head_dim):
    scale = 1.0 / (head_dim**0.5)
    q_ref = q.transpose(1, 2).float()
    k_ref = k.transpose(1, 2).float()
    v_ref = v.transpose(1, 2).float()
    scores = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v_ref).transpose(1, 2)


def cosine_sim(a, b):
    """Cosine similarity between two tensors (scalar)."""
    return torch.nn.functional.cosine_similarity(
        a.float().flatten().unsqueeze(0), b.float().flatten().unsqueeze(0)
    ).item()


def count_tiles(ranges, kBlockN):
    """Count tiles covered by [start0, end0, start1, end1, ...] ranges."""
    return sum(LiteAttention.ceil_div(ranges[i + 1], kBlockN) - ranges[i] // kBlockN for i in range(0, len(ranges), 2))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=HEAD_DIMS, ids=[f"d{d}" for d in HEAD_DIMS])
def head_dim(request):
    return request.param


@pytest.fixture
def qkv(head_dim):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    return generate_test_tensors(BATCH, SEQ_LEN, HEADS, head_dim)


@pytest.fixture
def qkv_short(head_dim):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    return generate_test_tensors(BATCH, min(6143, SEQ_LEN), HEADS, head_dim)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_int8", [False, True], ids=["bf16", "int8"])
def test_skip_all(qkv, use_int8):
    """threshold=inf: all tiles skipped except one range of length 1."""
    q, k, v = qkv
    attn = LiteAttention(use_int8=use_int8, threshold=float("inf"))
    warmup(attn, q, k, v)

    skip_list = attn._skip_list[attn._phase, : q.shape[0]]
    assert (skip_list[..., 0] == 2).all(), "Should contain exactly 1 range"
    diff = (skip_list[..., 1] - skip_list[..., 2]).abs()
    assert (diff == 1).all(), "Range length should be 1"


@pytest.mark.parametrize("use_int8", [False, True], ids=["bf16", "int8"])
def test_skip_nothing(qkv, use_int8):
    """threshold=-inf: no tiles skipped, read list matches initial list."""
    q, k, v = qkv
    attn = LiteAttention(use_int8=use_int8, threshold=float("-inf"))
    read_list_original, _ = attn._get_read_write_lists(q, v)
    read_list_original = read_list_original.clone()
    attn._phase = 0

    warmup(attn, q, k, v, 2)

    read_list = attn.read_list
    assert (read_list[..., 0] == 2).all(), "Should contain exactly 1 range"
    rl_range = read_list[..., 1:3]
    orig_range = read_list_original[..., 1:3]
    assert torch.equal(rl_range.min(dim=-1).values, orig_range.min(dim=-1).values), "Range min should match initial"
    assert torch.equal(rl_range.max(dim=-1).values, orig_range.max(dim=-1).values), "Range max should match initial"


@pytest.mark.parametrize("use_int8", [False, True], ids=["bf16", "int8"])
def test_softmax_lse_correctness(qkv_short, head_dim, use_int8):
    """Softmax LSE matches PyTorch reference."""
    q, k, v = qkv_short
    attn = LiteAttention(use_int8=use_int8, threshold=0.0)
    torch.cuda.synchronize()
    _, lse_lite = attn(q, k, v, return_softmax_lse=True)
    torch.cuda.synchronize()

    lse_ref = compute_reference_lse(q, k, head_dim)
    if lse_lite.dim() == 4 and lse_lite.shape[-1] == 1:
        lse_lite = lse_lite.squeeze(-1)

    tolerance = 0.1 if use_int8 else 0.001
    torch.testing.assert_close(lse_lite.float(), lse_ref, atol=tolerance, rtol=0)


@pytest.mark.parametrize("use_int8", [False, True], ids=["bf16", "int8"])
def test_rectangular_attention_correctness(head_dim, use_int8):
    """Rectangular attention (Lq != Lk) matches PyTorch reference."""
    tol_abs = 0.1 if use_int8 else 1e-2
    tol_cos = 0.99 if use_int8 else 0.999
    batch, q_len, k_len, heads = 1, 1024, 256, 4

    q, k, v = generate_rectangular_test_tensors(batch, q_len, k_len, heads, head_dim)
    scale = 1.0 / (head_dim**0.5)

    attn = LiteAttention(enable_skipping=False, use_int8=use_int8)
    torch.cuda.synchronize()
    output = attn(q, k, v, scale=scale)
    torch.cuda.synchronize()

    ref = compute_reference_attention_output(q, k, v, head_dim)
    torch.testing.assert_close(output.float(), ref, atol=tol_abs, rtol=0)
    assert cosine_sim(output, ref) >= tol_cos


@pytest.mark.parametrize("use_int8", [False, True], ids=["bf16", "int8"])
@pytest.mark.parametrize(("q_len", "k_len"), [(4096, 1024), (1024, 4096)], ids=["q4096_k1024", "q1024_k4096"])
def test_rectangular_attention_skipping_twice(head_dim, q_len, k_len, use_int8):
    """Two-pass rectangular attention with skipping produces valid skip lists."""
    batch, heads = 1, 4
    tile_dtype = torch.int8 if use_int8 else torch.bfloat16
    kBlockM, kBlockN = LiteAttention.get_MN(head_dim, tile_dtype, v_colmajor=False)

    device = "cuda"
    dtype = torch.bfloat16

    q_base_len = 2 * kBlockM + 1
    k_base_len = 4 * kBlockN
    assert q_len > q_base_len
    assert k_len > k_base_len

    # Deterministic per-head unit vectors
    base = torch.zeros(heads, head_dim, device=device, dtype=torch.float32)
    for h in range(heads):
        base[h, h % head_dim] = 1.0
    base = base.to(dtype)

    q_vec = (4.0 * base).view(1, 1, heads, head_dim)
    q_base = q_vec.repeat(batch, q_base_len, 1, 1).contiguous()

    # 4 key tiles: [+Q, -Q, -Q, +Q] — high tiles will be computed, low tiles skipped
    k_base = torch.empty(batch, k_base_len, heads, head_dim, device=device, dtype=dtype)
    k_base[:, :kBlockN] = q_vec
    k_base[:, kBlockN : 2 * kBlockN] = -q_vec
    k_base[:, 2 * kBlockN : 3 * kBlockN] = -q_vec
    k_base[:, 3 * kBlockN : 4 * kBlockN] = q_vec

    v_base = (0.1 * torch.randn(batch, k_base_len, heads, head_dim, device=device, dtype=dtype)).contiguous()
    q_extra = (0.1 * torch.randn(batch, q_len - q_base_len, heads, head_dim, device=device, dtype=dtype)).contiguous()
    k_extra = (0.1 * torch.randn(batch, k_len - k_base_len, heads, head_dim, device=device, dtype=dtype)).contiguous()
    v_extra = (0.1 * torch.randn(batch, k_len - k_base_len, heads, head_dim, device=device, dtype=dtype)).contiguous()

    q = torch.cat([q_base, q_extra], dim=1).contiguous()
    k = torch.cat([k_base, k_extra], dim=1).contiguous()
    v = torch.cat([v_base, v_extra], dim=1).contiguous()

    scale = 1.0 / (head_dim**0.5)
    attn = LiteAttention(enable_skipping=True, use_int8=use_int8, threshold=-1.0)

    for _pass_num in range(2):
        torch.cuda.synchronize()
        output = attn(q, k, v, scale=scale)
        torch.cuda.synchronize()

        rl = attn.read_list
        assert rl is not None
        assert rl[..., 0].max().item() > 0, "Skip list should not be empty"
        assert_skip_list_length_valid(rl)
        assert_no_empty_or_negative_ranges(rl)

    assert not torch.isnan(output).any()
    assert torch.isfinite(output.float()).all()

    pct = float(attn.calc_percentage(attn.read_list).item())
    assert 0.0 < pct < 1.0


@pytest.mark.parametrize("use_int8", [False, True], ids=["bf16", "int8"])
def test_stress(qkv, use_int8):
    """Skip percentage stays stable across repeated forward passes."""
    q, k, v = qkv
    attn = LiteAttention(use_int8=use_int8, threshold=0.0)
    warmup(attn, q, k, v, 2)

    percentage = attn.calc_percentage(attn.read_list).item()
    tol = 1e-4

    for _pass_num in range(10):
        torch.cuda.synchronize()
        attn(q, k, v)
        torch.cuda.synchronize()
        new_pct = attn.calc_percentage(attn.read_list).item()
        assert new_pct == pytest.approx(percentage, abs=tol)


@pytest.mark.parametrize("use_int8", [False, True], ids=["bf16", "int8"])
@pytest.mark.parametrize("case_fn", SKIP_CASES)
def test_must_skip_list(qkv, head_dim, use_int8, case_fn):
    """must_skip_list forces tiles to be skipped even when threshold=-inf."""
    q, k, v = qkv
    seq_len = k.shape[1]
    must_skip = case_fn(seq_len)
    element_type = torch.int8 if use_int8 else k.dtype
    _, kBlockN = LiteAttention.get_MN(head_dim, element_type)
    ktiles = LiteAttention.ceil_div(seq_len, kBlockN)

    attn = LiteAttention(use_int8=use_int8, threshold=-float("inf"))
    torch.cuda.synchronize()
    attn(q, k, v, must_skip_list=must_skip)
    torch.cuda.synchronize()

    expected = (ktiles - count_tiles(must_skip, kBlockN)) / ktiles
    actual = attn.calc_percentage(attn.read_list).item()
    assert actual == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("use_int8", [False, True], ids=["bf16", "int8"])
@pytest.mark.parametrize("case_fn", DO_CASES)
def test_must_do_list(qkv, head_dim, use_int8, case_fn):
    """must_do_list forces tiles to be computed even when threshold=inf."""
    q, k, v = qkv
    seq_len = k.shape[1]
    must_do = case_fn(seq_len)
    element_type = torch.int8 if use_int8 else k.dtype
    _, kBlockN = LiteAttention.get_MN(head_dim, element_type)
    ktiles = LiteAttention.ceil_div(seq_len, kBlockN)

    attn = LiteAttention(use_int8=use_int8, threshold=float("inf"))
    for _ in range(10):
        torch.cuda.synchronize()
        attn(q, k, v, must_do_list=must_do)
        torch.cuda.synchronize()

        expected = count_tiles(must_do, kBlockN) / ktiles
        actual = attn.calc_percentage(attn.read_list).item()
        assert actual == pytest.approx(expected, abs=0.01)


def test_int8_correctness(qkv_short, head_dim):
    """INT8 output matches BF16 output (no skipping)."""
    q, k, v = qkv_short

    tile_size_bf16 = LiteAttention.get_MN(head_dim, torch.bfloat16, is_skipable=False)
    tile_size_int8 = LiteAttention.get_MN(head_dim, torch.int8, is_skipable=False)
    if tile_size_bf16 != tile_size_int8:
        pytest.skip(f"Tile sizes differ (BF16: {tile_size_bf16}, INT8: {tile_size_int8})")

    scale = 1.0 / (head_dim**0.5)

    attn_bf16 = LiteAttention(enable_skipping=False, use_int8=False)
    torch.cuda.synchronize()
    out_bf16 = attn_bf16(q, k, v, scale=scale)
    torch.cuda.synchronize()

    attn_int8 = LiteAttention(enable_skipping=False, use_int8=True)
    torch.cuda.synchronize()
    out_int8 = attn_int8(q, k, v, scale=scale)
    torch.cuda.synchronize()

    torch.testing.assert_close(out_int8.float(), out_bf16.float(), atol=0.1, rtol=0)
    assert cosine_sim(out_int8, out_bf16) >= 0.99


def test_int8_with_skipping(qkv_short, head_dim):
    """INT8 with skipping matches BF16 with skipping."""
    q, k, v = qkv_short

    tile_size_bf16 = LiteAttention.get_MN(head_dim, torch.bfloat16, is_skipable=True)
    tile_size_int8 = LiteAttention.get_MN(head_dim, torch.int8, is_skipable=True)
    if tile_size_bf16 != tile_size_int8:
        pytest.skip(f"Tile sizes differ (BF16: {tile_size_bf16}, INT8: {tile_size_int8})")

    scale = 1.0 / (head_dim**0.5)

    attn_bf16 = LiteAttention(enable_skipping=True, use_int8=False, threshold=0.0)
    warmup(attn_bf16, q, k, v, num_iters=2)
    torch.cuda.synchronize()
    out_bf16 = attn_bf16(q, k, v, scale=scale)
    torch.cuda.synchronize()

    attn_int8 = LiteAttention(enable_skipping=True, use_int8=True, threshold=0.0)
    warmup(attn_int8, q, k, v, num_iters=2)
    torch.cuda.synchronize()
    out_int8 = attn_int8(q, k, v, scale=scale)
    torch.cuda.synchronize()

    torch.testing.assert_close(out_int8.float(), out_bf16.float(), atol=0.15, rtol=0)
    assert cosine_sim(out_int8, out_bf16) >= 0.98

    pct_bf16 = attn_bf16.calc_percentage(attn_bf16.read_list).item()
    pct_int8 = attn_int8.calc_percentage(attn_int8.read_list).item()
    assert pct_bf16 == pytest.approx(pct_int8, abs=0.05)


@pytest.mark.skip(reason="Disabled in original test runner")
def test_consistency(qkv):
    """Skip percentage never increases and skip lists stay valid across random inputs."""
    q, k, v = qkv
    attn = LiteAttention(threshold=0.0)
    percentage = float("inf")

    for _pass_num in range(10):
        q, k, v = generate_test_tensors(*q.shape)
        torch.cuda.synchronize()
        attn(q, k, v)
        torch.cuda.synchronize()

        skip_list = attn.read_list
        new_pct = attn.calc_percentage(skip_list).item()
        assert new_pct <= percentage, "Percentage should not increase"
        percentage = new_pct
        assert_skip_list_length_valid(skip_list)
        assert_no_empty_or_negative_ranges(skip_list)
