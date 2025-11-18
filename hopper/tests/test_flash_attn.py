import os
import math
import itertools

import pytest
import torch
import torch.nn.functional as F
from torch._C import parse_schema

from einops import rearrange, repeat
try:
    from flash_attn.layers.rotary import apply_rotary_emb
except ImportError:
    apply_rotary_emb = None

from padding import pad_input, unpad_input
from test_util import (
    attention_ref,
    generate_qkv,
    generate_random_padding_mask,
)

from flash_attn_interface import flash_attn_func, flash_attn_varlen_func, flash_attn_combine
from flash_attn_interface import flash_attn_with_kvcache, get_scheduler_metadata
from lite_attention import LiteAttention


DISABLE_BACKWARD = os.getenv("FLASH_ATTENTION_DISABLE_BACKWARD", "FALSE") == "TRUE"
DISABLE_SPLIT = os.getenv("FLASH_ATTENTION_DISABLE_SPLIT", "FALSE") == "TRUE"
DISABLE_PAGEDKV = os.getenv("FLASH_ATTENTION_DISABLE_PAGEDKV", "FALSE") == "TRUE"
DISABLE_APPENDKV = os.getenv("FLASH_ATTENTION_DISABLE_APPENDKV", "FALSE") == "TRUE"
DISABLE_LOCAL = os.getenv("FLASH_ATTENTION_DISABLE_LOCAL", "FALSE") == "TRUE"
DISABLE_SOFTCAP = os.getenv("FLASH_ATTENTION_DISABLE_SOFTCAP", "FALSE") == "TRUE"
DISABLE_PACKGQA = os.getenv("FLASH_ATTENTION_DISABLE_PACKGQA", "FALSE") == "TRUE"
DISABLE_FP16 = os.getenv("FLASH_ATTENTION_DISABLE_FP16", "FALSE") == "TRUE"
DISABLE_FP8 = os.getenv("FLASH_ATTENTION_DISABLE_FP8", "FALSE") == "TRUE" or torch.cuda.get_device_capability("cuda")[0] < 9
DISABLE_HDIM64 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM64", "FALSE") == "TRUE"
DISABLE_HDIM96 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM96", "FALSE") == "TRUE"
DISABLE_HDIM128 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM128", "FALSE") == "TRUE"
DISABLE_HDIM192 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM192", "FALSE") == "TRUE"
DISABLE_HDIM256 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM256", "FALSE") == "TRUE"

COMPILED_HDIMS = (
    []
    + ([64] if not DISABLE_HDIM64 else [])
    + ([96] if not DISABLE_HDIM96 else [])
    + ([128] if not DISABLE_HDIM128 else [])
    + ([192] if not DISABLE_HDIM192 else [])
    + ([256] if not DISABLE_HDIM256 else [])
)

@pytest.mark.parametrize("dtype", [torch.bfloat16] + ([torch.float16] if not DISABLE_FP16 else []) + ([torch.float8_e4m3fn] if not DISABLE_FP8 else []))
@pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("has_qv", [False])
@pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("softcap", [0.0] + ([15.0] if not DISABLE_SOFTCAP else []))
@pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("V_colmajor", [False, True])
@pytest.mark.parametrize("d", COMPILED_HDIMS)
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 1),
        (64, 128),
        (128, 192),
        (256, 256),
        (239, 1),
        (799, 3),
        (113, 203),
        (113, 128),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (384, 256),
        (640, 128),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (4096, 4096),
        (4224, 4224),
    ],
)
def test_lite_attn_output(seqlen_q, seqlen_k, d, causal, local, softcap, V_colmajor, deterministic, has_qv, mha_type, dtype):
    if V_colmajor and (seqlen_k % 16 != 0 or dtype != torch.float8_e4m3fn):
        pytest.skip("V_colmajor requires seqlen_k to be a multiple of 16 and dtype to be float8_e4m3fn")
    if has_qv and (d != 64 or dtype == torch.float8_e4m3fn):
        pytest.skip("Has Qv requires hdim 64 and dtype to be float16 or bfloat16 (not float8_e4m3fn)")
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    # batch_size = 40
    # nheads = 16
    batch_size = 9 if seqlen_k <= 2048 else 2
    # batch_size = 1
    nheads = 6
    # nheads = 1
    nheads_kv = nheads if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    dv_vals = [128, d] if d > 128 and d <= 192 else ([256, 512, d] if d <= 64 else [d])
    if dtype == torch.float8_e4m3fn:
        dv_vals = [d]
    if has_qv:
        dv_vals = [256, 512]
    attention_chunk_vals = [torch.randint(1, seqlen_k * 2, (1,)).item(), 0] if not DISABLE_LOCAL else [0]
    for dv, attention_chunk in itertools.product(dv_vals, attention_chunk_vals):
        lite_attn = LiteAttention(enable_skipping=True, threshold=-3.0)

        print(f"{dv = }, {attention_chunk = }")
        # q_ref = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref)
        q_ref = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref)
        if softcap > 0.0:
            # Ensure the values of qk are at least within softcap range.
            q_ref = (q_ref * softcap / 4)
        q_ref = q_ref.to(dtype).to(dtype_ref).requires_grad_()
        k_ref = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()
        v_ref = torch.randn(batch_size, seqlen_k, nheads_kv, dv, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()
        if has_qv:
            qv_ref = torch.randn(batch_size, seqlen_q, nheads, dv, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
        else:
            qv_ref = None

# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("dtype", [torch.bfloat16] + ([torch.float16] if not DISABLE_FP16 else []) + ([torch.float8_e4m3fn] if not DISABLE_FP8 else []))
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("has_qv", [False, True])
# @pytest.mark.parametrize("has_qv", [True])
# @pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("softcap", [0.0] + ([15.0] if not DISABLE_SOFTCAP else []))
# @pytest.mark.parametrize("softcap", [0.0])
@pytest.mark.parametrize("local", [False] + ([True] if not DISABLE_LOCAL else []))
# @pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [True])
# @pytest.mark.parametrize("V_colmajor", [False, True])
@pytest.mark.parametrize("V_colmajor", [False])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192, 256])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize("d", [64, 128, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128])
# @pytest.mark.parametrize("d", [64, 96, 128, 192])
@pytest.mark.parametrize("d", COMPILED_HDIMS)
# @pytest.mark.parametrize("d", [64])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 1),
        (64, 128),
        (128, 192),
        (256, 256),
        (239, 1),
        (799, 3),
        (113, 203),
        (113, 128),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (384, 256),
        (640, 128),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (4096, 4096),
        (4224, 4224),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(128, 128)])
def test_flash_attn_output(
        seqlen_q, seqlen_k, d, causal, local, softcap, V_colmajor, deterministic, has_qv, mha_type, dtype
):
    if V_colmajor and (seqlen_k % 16 != 0 or dtype != torch.float8_e4m3fn):
        pytest.skip("V_colmajor requires seqlen_k to be a multiple of 16 and dtype to be float8_e4m3fn")
    if has_qv and (d != 64 or dtype == torch.float8_e4m3fn):
        pytest.skip("Has Qv requires hdim 64 and dtype to be float16 or bfloat16 (not float8_e4m3fn)")
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    # batch_size = 40
    # nheads = 16
    batch_size = 9 if seqlen_k <= 2048 else 2
    # batch_size = 1
    nheads = 6
    # nheads = 1
    nheads_kv = nheads if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    dv_vals = [128, d] if d > 128 and d <= 192 else ([256, 512, d] if d <= 64 else [d])
    if dtype == torch.float8_e4m3fn:
        dv_vals = [d]
    if has_qv:
        dv_vals = [256, 512]
    attention_chunk_vals = [torch.randint(1, seqlen_k * 2, (1,)).item(), 0] if not DISABLE_LOCAL else [0]
    for dv, attention_chunk in itertools.product(dv_vals, attention_chunk_vals):
        print(f"{dv = }, {attention_chunk = }")
        q_ref = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref)
        if softcap > 0.0:
            # Ensure the values of qk are at least within softcap range.
            q_ref = (q_ref * softcap / 4)
        q_ref = q_ref.to(dtype).to(dtype_ref).requires_grad_()
        k_ref = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()
        v_ref = torch.randn(batch_size, seqlen_k, nheads_kv, dv, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()
        if has_qv:
            qv_ref = torch.randn(batch_size, seqlen_q, nheads, dv, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
        else:
            qv_ref = None
        # Put window_size after QKV randn so that window_size changes from test to test
        window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,)).tolist()
        # window_size = (-1, -1) if not local else (16, 0)
        if dtype == torch.float8_e4m3fn:
            q_descale, k_descale, v_descale = [torch.rand(batch_size, nheads_kv, device=device, dtype=torch.float32) * 2 for _ in range(3)]
        else:
            q_descale, k_descale, v_descale = None, None, None
        q, k, v = [x.detach().to(dtype).requires_grad_() for x in (q_ref, k_ref, v_ref)]
        qv = qv_ref.detach().to(dtype).requires_grad_() if has_qv else None
        if V_colmajor:
            v = rearrange(rearrange(v.detach(), "b s h d -> b h d s").contiguous(), "b h d s -> b s h d").requires_grad_()
        out_ref, attn_ref = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            None,
            None,
            causal=causal,
            qv=qv_ref,
            q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
            window_size=window_size,
            attention_chunk=attention_chunk,
            softcap=softcap
        )
        out_pt, attn_pt = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            None,
            None,
            causal=causal,
            qv=qv_ref,
            q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
            window_size=window_size,
            attention_chunk=attention_chunk,
            softcap=softcap,
            upcast=False,
            reorder_ops=True,
            intermediate_dtype=dtype if dtype == torch.float8_e4m3fn else None,
        )

        # qk = torch.einsum('bshd,bthd->bhst', q_ref, k_ref).float()
        # if qv is not None:
        #     qk += torch.einsum('bshd,bthd->bhst', qv_ref, v_ref).float()
        # m = qk.amax(-1, keepdim=True)
        # s_tmp = torch.exp((qk - m) / math.sqrt(d))
        # exp_sum = s_tmp.sum(-1)
        # qk = torch.einsum('bthd,bshd->bhts', q_ref.float() / math.sqrt(d), k_ref.float())
        # lse_ref = torch.logsumexp(qk, dim=-1)

        # Numerical error if we just do any arithmetic on out_ref
        fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
        rtol = 2 if softcap == 0.0 else 3

        print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
        print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
        pack_gqa_vals = [False, True] if not DISABLE_PACKGQA else [False]
        num_splits_vals = [1, 3] if not DISABLE_SPLIT else [1]
        for pack_gqa, num_splits in itertools.product(pack_gqa_vals, num_splits_vals):
            print(f"{pack_gqa = }, {num_splits = }")
            out = flash_attn_func(
                q,
                k,
                v,
                causal=causal,
                qv=qv,
                q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
                window_size=window_size,
                attention_chunk=attention_chunk,
                softcap=softcap,
                pack_gqa=pack_gqa,
                num_splits=num_splits
            )
            print(f"Output max diff: {(out - out_ref).abs().max().item()}")
            print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
            # if not causal:
            #     print(f"LSE max diff: {(lse - lse_ref).abs().max().item()}")
            # breakpoint()

            # Check that FlashAttention's numerical error is at most twice the numerical error
            # of a Pytorch implementation.
            assert (out - out_ref).abs().max().item() <= rtol * (out_pt - out_ref).abs().max().item() + fwd_atol

        if (
            not DISABLE_BACKWARD 
            and dtype != torch.float8_e4m3fn 
            and not V_colmajor 
            and not has_qv
            and not dv > 256
            and not attention_chunk != 0
        ):
            g = torch.randn_like(out)
            do_o = ((g.float() * out.float()).sum(-1)).transpose(1, 2)
            # import flash_attn_3_cuda
            # dq, dk, dv, softmax_d, dq_accum, dk_accum, dv_accum = flash_attn_3_cuda.bwd(
            #     g,
            #     q,
            #     k,
            #     v,
            #     out,
            #     lse,
            #     None,
            #     None,
            #     None,
            #     d ** (-0.5),
            #     causal,
            #     window_size[0], window_size[1],
            #     softcap,
            #     deterministic,
            #     0,  # sm_margin
            # )
            dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
            # print(f"dO_O max diff: {(softmax_d - do_o).abs().max().item()}")
            # assert (softmax_d - do_o).abs().max().item() <= 1e-5
            # assert dq_accum.abs().max().item() == 0.0

            # dS = torch.einsum('bthd,bshd->bhts', g.float(), v.float())
            # P = torch.softmax(qk, -1)
            # dP = P * (dS - do_o.transpose(1, 2).unsqueeze(1))
            # dQ = torch.einsum('bhts,bshd->bthd', dP, k.float())
            # dV = torch.einsum('bhts,bthd->bshd', P, g.float())
            # dK = torch.einsum('bhts,bthd->bshd', dP, q.float())

            # dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
            dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), g)
            dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), g)
            print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
            print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
            print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
            print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
            print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
            print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
            print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
            print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
            print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
            print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
            print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
            print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")
            # breakpoint()
            dq_atol = 2 * (dq_ref + 0.3 - 0.3 - dq_ref).abs().max().item() + (0 if softcap == 0 else 3e-4)
            assert (dq - dq_ref).abs().max().item() <= rtol * (dq_pt - dq_ref).abs().max().item() + dq_atol
            dk_atol = 2 * (dk_ref + 0.3 - 0.3 - dk_ref).abs().max().item() + (0 if softcap == 0 else 3e-4)
            assert (dk - dk_ref).abs().max().item() <= rtol * (dk_pt - dk_ref).abs().max().item() + dk_atol
            dv_atol = 2 * (dv_ref + 0.3 - 0.3 - dv_ref).abs().max().item() + (0 if softcap == 0 else 3e-4)
            assert (dv - dv_ref).abs().max().item() <= rtol * (dv_pt - dv_ref).abs().max().item() + dv_atol


# @pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128])
# @pytest.mark.parametrize('d', [32, 56, 64, 80, 96, 128])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [80])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 239),
        (239, 1),
        (3, 799),
        (799, 3),
        (1024, 128),
        (97, 97),
        (128, 128),
        (200, 200),
        (256, 256),
        (257, 257),
        (384, 384),
        (512, 512),
        (768, 768),
        (1024, 1024),
        (2048, 2048),
    ],
)
def test_flash_attn_race_condition(seqlen_q, seqlen_k, d, causal, dtype):
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    # Simulate under memory load
    dummy = torch.empty(70 * 1024 ** 3, dtype=torch.uint8, device=device)
    batch_size = 60  # Sometimes we need large batch size for the race conditions to trigger
    nheads = 4
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    torch.random.manual_seed(42)
    out0 = flash_attn_func(q, k, v, causal=causal)
    g = torch.randn_like(out0)
    dq0, dk0, dv0 = torch.autograd.grad(out0, (q, k, v), g)
    # Numerical error if we just do any arithmetic on dq
    dq_atol = 2 * ((dq0 + 0.3 - 0.3) - dq0).abs().max().item()

    for i in range(1000):
        torch.random.manual_seed(42)
        out = flash_attn_func(q, k, v, causal=causal)
        assert torch.equal(out, out0)
        # assert torch.equal(lse, lse0)

        dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
        dq_equal = torch.allclose(dq, dq0, atol=dq_atol)
        if not dq_equal:
            print(f"Iter {i}, {dq_atol = }, dQ max diff: {(dq - dq0).abs().max().item()}")
            # breakpoint()
        assert torch.equal(dv, dv0)
        assert torch.equal(dk, dk0)
        assert dq_equal