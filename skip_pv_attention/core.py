
import torch
from .quant_per_block import per_block_int8
from .utils import hyperparameter_check
from .kernel_skip_pv import forward as skip_pv_attn_forward
from .kernel_ref import forward as ref_attn_forward
from torch.nn.functional import scaled_dot_product_attention

TIME_STEP_INTERVAL = 4
FORCE_FP16 = True

def skip_pv_attention(q, k, v, is_causal=False, smooth_k=True, pvthreshd=20, BLOCK_M=128, BLOCK_N=64, **kwargs):
    assert q.size(-2)>=128, "seq_len should be not less than 128."
    
    # print(f"pvthreshd: {pvthreshd}")

    torch.cuda.set_device(v.device)

    dtype = q.dtype
    assert dtype == torch.bfloat16
        # if dtype == torch.float32 or dtype == torch.float16:
        #     q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
        # else:
        #     q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    if smooth_k:
        k = k - k.mean(dim=-2, keepdim=True)

    k_block_indices = torch.ones((q.size(0), q.size(1), (q.size(2)+1)//128, (k.size(2)+1)//64), device=q.device, dtype=torch.int8)

    headdim = q.size(-1)

    assert headdim in [64, 128], "headdim should be in [64, 96, 128]."

    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, BLKQ=BLOCK_M, BLKK=BLOCK_N)
    pvthreshd = hyperparameter_check(pvthreshd, q.size(-3), q.device)

    o, skip_ratio_per_head = skip_pv_attn_forward(
        q_int8, k_int8, k_block_indices, v, q_scale, k_scale,
        pvthreshd, is_causal=is_causal, tensor_layout="HND", output_dtype=dtype, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, **kwargs
    )
    
    return o, skip_ratio_per_head


def skip_pv_attention_ref(q, k, v, is_causal=False, smooth_k=True, pvthreshd=20, BLOCK_M=128, BLOCK_N=64, **kwargs):
    assert q.size(-2)>=128, "seq_len should be not less than 128."
    
    # print(f"pvthreshd: {pvthreshd}")

    torch.cuda.set_device(v.device)

    dtype = q.dtype
    assert dtype == torch.bfloat16
    # if dtype == torch.float32 or dtype == torch.float16:
    #     q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    # else:
    #     q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    if smooth_k:
        k = k - k.mean(dim=-2, keepdim=True)

    k_block_indices = torch.ones((q.size(0), q.size(1), (q.size(2)+1)//128, (k.size(2)+1)//64), device=q.device, dtype=torch.int8)

    headdim = q.size(-1)

    assert headdim in [64, 128], "headdim should be in [64, 96, 128]."

    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, BLKQ=BLOCK_M, BLKK=BLOCK_N)
    pvthreshd = hyperparameter_check(pvthreshd, q.size(-3), q.device)

    o, skip_ratio_per_head = ref_attn_forward(
        q_int8, k_int8, k_block_indices, v, q_scale, k_scale,
        pvthreshd, is_causal=is_causal, tensor_layout="HND", output_dtype=dtype, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, **kwargs
    )
    
    return o, skip_ratio_per_head

attention_dict = {
    "skip_pv": skip_pv_attention,
    "sdpa": scaled_dot_product_attention,
    "ref": skip_pv_attention_ref,
}

def attention(q, k, v, kernel="default", **kwargs):
    # print(f"pvthreshd: {kwargs.get('pvthreshd', None)}")
    # print("torch.max(v), torch.min(v)", torch.max(v), torch.min(v))
    # print("torch.max(q), torch.min(q)", torch.max(q), torch.min(q))
    # print("torch.max(k), torch.min(k)", torch.max(k), torch.min(k))
    # if FORCE_FP16:
    #     if v.dtype in [torch.float32, torch.bfloat16]:
    #         v = v.to(torch.float16)
    # print("v.dtype", v.dtype, "q.dtype", q.dtype, "k.dtype", k.dtype)
    return attention_dict[kernel](q, k, v, **kwargs)
