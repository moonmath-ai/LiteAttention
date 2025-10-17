"""
Copyright (c) 2025 by SpargeAttn team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch, math
import triton
import triton.language as tl

global_skip_count = 0
global_operation_count = 0

HEAD_WINDOW_SIZE = 2
MIDDLE_WINDLE_SIZE = 2

@triton.jit
def _attn_fwd_inner(acc, l_i, old_m, q, q_scale, kv_len,
                    K_ptrs, K_bid_ptr, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
                    pvthreshd, start_m,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                    skip_count,  # Remove tl.constexpr
                    PREFIX_WINDOW_SIZE: tl.constexpr, MIDDLE_WINDOW_SIZE: tl.constexpr,
                    ):
    # Base pointers (starting at the 0th N-block), offset by start_n subsequently
    K_ptrs_base = K_ptrs
    V_ptrs_base = V_ptrs
    K_scale_ptr_base = K_scale_ptr

    # Use the center row of the current row-block as reference to estimate diagonal column block index
    m_center = start_m * BLOCK_M + (BLOCK_M // 2)
    diag_block_idx = m_center // BLOCK_N

    # Set iteration window based on STAGE
    lo, hi = 0, kv_len
    
    # First stage: Prioritize traversing diagonal window blocks (traverse all column blocks within this window range in ascending order)
    for start_n in range(lo, hi, BLOCK_N):
        blk_idx = start_n // BLOCK_N
        is_in_middle = tl.abs(blk_idx - diag_block_idx) < MIDDLE_WINDOW_SIZE
        if is_in_middle:
            K_ptrs_cur = K_ptrs_base + start_n * stride_kn
            V_ptrs_cur = V_ptrs_base + start_n * stride_vn
            K_scale_ptr_cur = K_scale_ptr_base + blk_idx

            k_mask = offs_n[None, :] < (kv_len - start_n)
            k = tl.load(K_ptrs_cur, mask=k_mask)
            k_scale = tl.load(K_scale_ptr_cur)
            qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale
            local_m = tl.max(qk, 1)
            new_m = tl.maximum(old_m, local_m)
            qk = qk - new_m[:, None]
            if pvthreshd == -1 or tl.min(new_m - local_m) < pvthreshd:
                p = tl.math.exp2(qk)
                l_ij = tl.sum(p, 1)
                alpha = tl.math.exp2(old_m - new_m)
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None]
                v = tl.load(V_ptrs_cur, mask=offs_n[:, None] < (kv_len - start_n))
                p = p.to(tl.bfloat16)
                acc += tl.dot(p, v).to(tl.bfloat16)
                old_m = new_m
            else:
                skip_count += 1

    # Second stage: Traverse prefix blocks (excluding blocks already in diagonal window)
    for start_n in range(lo, hi, BLOCK_N):
        blk_idx = start_n // BLOCK_N
        is_in_middle = tl.abs(blk_idx - diag_block_idx) < MIDDLE_WINDOW_SIZE
        is_prefix = blk_idx < PREFIX_WINDOW_SIZE
        if (not is_in_middle) and is_prefix:
            K_ptrs_cur = K_ptrs_base + start_n * stride_kn
            V_ptrs_cur = V_ptrs_base + start_n * stride_vn
            K_scale_ptr_cur = K_scale_ptr_base + blk_idx

            k_mask = offs_n[None, :] < (kv_len - start_n)
            k = tl.load(K_ptrs_cur, mask=k_mask)
            k_scale = tl.load(K_scale_ptr_cur)
            qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale
            local_m = tl.max(qk, 1)
            new_m = tl.maximum(old_m, local_m)
            qk = qk - new_m[:, None]
            if pvthreshd == -1 or tl.min(new_m - local_m) < pvthreshd:
                p = tl.math.exp2(qk)
                l_ij = tl.sum(p, 1)
                alpha = tl.math.exp2(old_m - new_m)
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None]
                v = tl.load(V_ptrs_cur, mask=offs_n[:, None] < (kv_len - start_n))
                p = p.to(tl.bfloat16)
                acc += tl.dot(p, v).to(tl.bfloat16)
                old_m = new_m
            else:
                skip_count += 1

    # Third stage: Traverse remaining blocks
    for start_n in range(lo, hi, BLOCK_N):
        blk_idx = start_n // BLOCK_N
        is_in_middle = tl.abs(blk_idx - diag_block_idx) < MIDDLE_WINDOW_SIZE
        is_prefix = blk_idx < PREFIX_WINDOW_SIZE
        if (not is_in_middle) and (not is_prefix):
            K_ptrs_cur = K_ptrs_base + start_n * stride_kn
            V_ptrs_cur = V_ptrs_base + start_n * stride_vn
            K_scale_ptr_cur = K_scale_ptr_base + blk_idx

            k_mask = offs_n[None, :] < (kv_len - start_n)
            k = tl.load(K_ptrs_cur, mask=k_mask)
            k_scale = tl.load(K_scale_ptr_cur)
            qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale
            local_m = tl.max(qk, 1)
            new_m = tl.maximum(old_m, local_m)
            qk = qk - new_m[:, None]
            if pvthreshd == -1 or tl.min(new_m - local_m) < pvthreshd:
                p = tl.math.exp2(qk)
                l_ij = tl.sum(p, 1)
                alpha = tl.math.exp2(old_m - new_m)
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None]
                v = tl.load(V_ptrs_cur, mask=offs_n[:, None] < (kv_len - start_n))
                p = p.to(tl.bfloat16)
                acc += tl.dot(p, v).to(tl.bfloat16)
                old_m = new_m
            else:
                skip_count += 1

    return acc, l_i, old_m, skip_count

@triton.jit
def _attn_fwd(Q, K, K_blkid, V, Q_scale, K_scale, PVThreshd, Out,  
              stride_qz, stride_qh, stride_qn,
              stride_kz, stride_kh, stride_kn,  
              stride_vz, stride_vh, stride_vn,  
              stride_oz, stride_oh, stride_on,  
              stride_kbidq, stride_kbidk,
              qo_len, kv_len, H:tl.constexpr, num_kv_groups:tl.constexpr, 
              HEAD_DIM: tl.constexpr,  
              BLOCK_M: tl.constexpr,  
              BLOCK_N: tl.constexpr,  
              STAGE: tl.constexpr, # suppose to be 1
              Skip_counts,  # Remove tl.constexpr
              PREFIX_WINDOW_SIZE: tl.constexpr, MIDDLE_WINDOW_SIZE: tl.constexpr
              ):
    start_m = tl.program_id(0)
    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)
    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)
    k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N)  
    k_bid_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * stride_kbidq
    pvthreshd = tl.load(PVThreshd+off_h)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None] 
    K_scale_ptr = K_scale + k_scale_offset
    K_bid_ptr = K_blkid + k_bid_offset + start_m * stride_kbidk 
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    skip_count = 0
    q = tl.load(Q_ptrs, mask = offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale_ptr)
    acc, l_i, m_i, skip_count = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, kv_len, K_ptrs, K_bid_ptr, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
                                    pvthreshd, start_m,  
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  
                                    4 - STAGE, offs_m, offs_n, skip_count,
                                    PREFIX_WINDOW_SIZE, MIDDLE_WINDOW_SIZE
                                    )
    if STAGE != 1:
        acc, l_i, _, skip_count = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, kv_len, K_ptrs, K_bid_ptr, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
                                        pvthreshd, start_m,  
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  
                                        2, offs_m, offs_n, skip_count,
                                        PREFIX_WINDOW_SIZE, MIDDLE_WINDOW_SIZE
                                        )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < qo_len))

    # Map batch and head together to one dimension (B*H)
    skip_count_ptr = Skip_counts + (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M) + start_m
    tl.store(skip_count_ptr, skip_count)


def forward(q, k, k_block_id, v, q_scale, k_scale, pvthreshd, is_causal=False, tensor_layout="HND", output_dtype=torch.float16, BLOCK_M=128, BLOCK_N=64,
            prefix_window_size: int = 2, middle_window_size: int = 2):
    stage = 3 if is_causal else 1
    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)
    skip_counts = torch.zeros((q.shape[0] * q.shape[1], triton.cdiv(q.shape[2], BLOCK_M)), 
                             device=q.device, dtype=torch.int32)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")
    
    assert qo_len == kv_len, "qo_len and kv_len must be equal for causal attention"

    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv

    grid = (triton.cdiv(qo_len, BLOCK_M), h_qo, b   )
    _attn_fwd[grid](
        q, k, k_block_id, v, q_scale, k_scale, pvthreshd, o,  
        stride_bz_q, stride_h_q, stride_seq_q, 
        stride_bz_k, stride_h_k, stride_seq_k,  
        stride_bz_v, stride_h_v, stride_seq_v,  
        stride_bz_o, stride_h_o, stride_seq_o,
        k_block_id.stride(1), k_block_id.stride(2),
        qo_len, kv_len,
        h_qo, num_kv_groups,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K,  
        STAGE=stage,  
        num_warps=4 if head_dim == 64 else 8,
        num_stages=4,
        Skip_counts=skip_counts,
        PREFIX_WINDOW_SIZE=prefix_window_size,
        MIDDLE_WINDOW_SIZE=middle_window_size
        )
    # Calculate per-head skip ratio
    num_m_blocks = triton.cdiv(q.shape[2], BLOCK_M)
    num_n_blocks = triton.cdiv(k.shape[2], BLOCK_N)
    per_head_skips = skip_counts.sum(dim=1).to(torch.float32)  # [B*H]
    per_head_total_ops = num_m_blocks * num_n_blocks
    skip_ratio_per_head = (per_head_skips / per_head_total_ops) * 100.0

    # Maintain global statistics for existing print functions to use
    total_skips = per_head_skips.sum().item()
    total_operations = (q.shape[0] * q.shape[1]) * per_head_total_ops
    global global_skip_count
    global global_operation_count
    global_skip_count += total_skips
    global_operation_count += total_operations

    return o, skip_ratio_per_head

def print_skip_ratio():
    global global_skip_count
    global global_operation_count
    skip_ratio = global_skip_count / global_operation_count * 100 if global_operation_count > 0 else 0
    print(f"Skip ratio: {skip_ratio:.2f}%")
    
def reset_skip_ratio():
    global global_skip_count
    global global_operation_count
    global_skip_count = 0
    global_operation_count = 0