// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026, LiteAttention contributors.
//
// Minimal CK fmha_fwd wrapper for LiteAttention on AMD MI300X.
// Only supports: batch mode, bf16, no bias/alibi/dropout/fp8/varlen/kvcache.

#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

#include "fmha_fwd.hpp"
#include "mask.hpp"

// ---------------------------------------------------------------------------
// lite_attn_fwd — the only entry point
// ---------------------------------------------------------------------------
std::vector<at::Tensor>
lite_attn_fwd(at::Tensor q,         // [B, S_q, H_q, D]
              at::Tensor k,         // [B, S_k, H_k, D]
              at::Tensor v,         // [B, S_k, H_k, D]
              float softmax_scale,
              bool is_causal,
              std::optional<at::Tensor> skip_read,      // flat int16 buffer
              std::optional<at::Tensor> skip_write,     // flat int16 buffer
              float skip_threshold,
              bool skip_reverse_list,
              int8_t skip_phase)
{
    // --- basic checks ---
    TORCH_CHECK(q.dtype() == at::ScalarType::BFloat16, "q must be bf16");
    TORCH_CHECK(k.dtype() == at::ScalarType::BFloat16, "k must be bf16");
    TORCH_CHECK(v.dtype() == at::ScalarType::BFloat16, "v must be bf16");
    TORCH_CHECK(q.stride(-1) == 1, "q must have contiguous last dim");
    TORCH_CHECK(k.stride(-1) == 1, "k must have contiguous last dim");
    TORCH_CHECK(v.stride(-1) == 1, "v must have contiguous last dim");

    const int B      = q.size(0);
    const int S_q    = q.size(1);
    const int H_q    = q.size(2);
    const int D      = q.size(3);
    const int S_k    = k.size(1);
    const int H_k    = k.size(2);

    TORCH_CHECK(B > 0, "batch size must be positive");
    TORCH_CHECK(D <= 256, "head dim must be <= 256");
    TORCH_CHECK(D % 8 == 0, "head dim must be a multiple of 8");
    TORCH_CHECK(H_q % H_k == 0, "H_q must be divisible by H_k");

    // --- allocate output + lse ---
    auto opts = q.options();
    at::Tensor out = torch::empty({B, S_q, H_q, D}, opts);
    at::Tensor softmax_lse = torch::empty({B, H_q, S_q}, opts.dtype(torch::kFloat32));

    const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard{q.device()};

    if (S_k == 0) {
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
        return {out, softmax_lse};
    }

    // --- mask ---
    mask_info mask;
    if (is_causal) {
        std::string mask_id = "b:-1,0,0";
        mask = mask_info::decode(mask_id, S_q, S_k);
    } else {
        mask = mask_info::decode("0", S_q, S_k);
    }

    // --- build fmha_fwd_traits (runtime dispatch key) ---
    bool is_lite = skip_read.has_value() && skip_read.value().numel() > 0;

    fmha_fwd_traits traits{
        /*hdim_q=*/D,
        /*hdim_v=*/D,
        /*data_type=*/std::string("bf16"),
        /*is_group_mode=*/false,
        /*is_v_rowmajor=*/true,
        /*has_logits_soft_cap=*/false,
        /*mask_type=*/mask.type,
        /*bias_type=*/bias_enum::no_bias,
        /*has_lse=*/!is_lite,  // lite pipeline doesn't support LSE output
        /*has_dropout=*/false,
        /*qscale_type=*/quant_scale_enum::no_scale,
        /*skip_min_seqlen_q=*/false,
        /*has_sink=*/false,
        /*is_lite=*/is_lite,
    };

    // --- build fmha_fwd_args ---
    // Dummy drop_seed_offset (no dropout)
    auto drop_seed_offset = std::pair<uint64_t, uint64_t>{0, 0};

    fmha_fwd_args args{
        /*q_ptr=*/q.data_ptr(),
        /*k_ptr=*/k.data_ptr(),
        /*v_ptr=*/v.data_ptr(),
        /*bias_ptr=*/nullptr,
        /*q_descale_ptr=*/nullptr,
        /*k_descale_ptr=*/nullptr,
        /*v_descale_ptr=*/nullptr,
        /*rand_val_ptr=*/nullptr,
        /*lse_ptr=*/is_lite ? nullptr : softmax_lse.data_ptr(),
        /*o_ptr=*/out.data_ptr(),
        /*seqstart_q_ptr=*/nullptr,
        /*seqstart_k_ptr=*/nullptr,
        /*seqlen_q_ptr=*/nullptr,
        /*seqlen_k_ptr=*/nullptr,
        /*cu_seqlen_q_ptr=*/nullptr,
        /*cu_seqlen_k_ptr=*/nullptr,
        /*block_scale_seqstart_q_ptr=*/nullptr,
        /*block_scale_seqstart_k_ptr=*/nullptr,
        /*seqstart_v_scale_ptr=*/nullptr,
        /*sink_ptr=*/nullptr,
        /*seqlen_q=*/S_q,
        /*seqlen_k=*/S_k,
        /*batch=*/B,
        /*max_seqlen_q=*/S_q,
        /*hdim_q=*/D,
        /*hdim_v=*/D,
        /*nhead_q=*/H_q,
        /*nhead_k=*/H_k,
        /*scale_s=*/softmax_scale,
        /*logits_soft_cap=*/0.0f,
        // strides: q/k/v/o are [B, S, H, D] contiguous-last-dim
        /*stride_q=*/q.stride(1),
        /*stride_k=*/k.stride(1),
        /*stride_v=*/v.stride(1),
        /*stride_bias=*/0,
        /*stride_randval=*/0,
        /*stride_o=*/out.stride(1),
        /*stride_q_descale=*/0,
        /*stride_k_descale=*/0,
        /*stride_v_descale=*/0,
        /*nhead_stride_q=*/q.stride(2),
        /*nhead_stride_k=*/k.stride(2),
        /*nhead_stride_v=*/v.stride(2),
        /*nhead_stride_bias=*/0,
        /*nhead_stride_randval=*/0,
        /*nhead_stride_lse=*/softmax_lse.stride(1),
        /*nhead_stride_o=*/out.stride(2),
        /*nhead_stride_q_descale=*/0,
        /*nhead_stride_k_descale=*/0,
        /*nhead_stride_v_descale=*/0,
        /*batch_stride_q=*/q.stride(0),
        /*batch_stride_k=*/k.stride(0),
        /*batch_stride_v=*/v.stride(0),
        /*batch_stride_bias=*/0,
        /*batch_stride_randval=*/0,
        /*batch_stride_lse=*/softmax_lse.stride(0),
        /*batch_stride_o=*/out.stride(0),
        /*batch_stride_q_descale=*/0,
        /*batch_stride_k_descale=*/0,
        /*batch_stride_v_descale=*/0,
        /*window_size_left=*/mask.left,
        /*window_size_right=*/mask.right,
        /*sink_size=*/mask.sink,
        /*mask_type=*/static_cast<ck_tile::index_t>(mask.type),
        /*min_seqlen_q=*/0,
        /*p_drop=*/0.0f,
        /*s_randval=*/false,
        /*drop_seed_offset=*/drop_seed_offset,
        /*block_scale_size_q=*/128,
        /*block_scale_size_kv=*/128,
    };

    // --- skip list strides ---
    const int kBlockM = 128;
    const int kBlockN = (D <= 64) ? 64 : 128;
    const int qtiles = (S_q + kBlockM - 1) / kBlockM;
    const int ktiles = (S_k + kBlockN - 1) / kBlockN;

    args.attn_read_list_ptr = skip_read.has_value()
        ? static_cast<const void*>(skip_read.value().data_ptr<int16_t>())
        : nullptr;
    args.attn_write_list_ptr = skip_write.has_value()
        ? static_cast<void*>(skip_write.value().data_ptr<int16_t>())
        : nullptr;
    args.attn_list_stride_q = static_cast<ck_tile::index_t>(ktiles + 2);
    args.attn_list_stride_h = static_cast<ck_tile::index_t>(qtiles) * args.attn_list_stride_q;
    args.attn_list_stride_b = H_q * args.attn_list_stride_h;
    args.threshold = skip_threshold;
    args.reverse_skip_list = skip_reverse_list;
    args.reverse_skip_list_phase = skip_phase;

    // --- launch ---
    auto stream = at::hip::getCurrentHIPStream();
    ck_tile::stream_config stream_config{stream};

    float t = fmha_fwd(traits, args, stream_config);
    TORCH_CHECK(t >= 0, "fmha_fwd dispatch failed (no matching kernel)");

    return {out, softmax_lse};
}

// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lite_attn_fwd", &lite_attn_fwd,
          "LiteAttention forward (CK fmha with skip lists)",
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("softmax_scale"),
          py::arg("is_causal"),
          py::arg("skip_read") = py::none(),
          py::arg("skip_write") = py::none(),
          py::arg("skip_threshold") = -1e30f,
          py::arg("skip_reverse_list") = false,
          py::arg("skip_phase") = static_cast<int8_t>(-1));
}
