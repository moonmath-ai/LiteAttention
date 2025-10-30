import torch
from lite_attention import LiteAttention

torch.manual_seed(0)
torch.cuda.manual_seed(0)

for head_dim in [32, 64, 96, 128, 192, 256]:
    q = torch.randn(2, 5000, 32, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 5000, 32, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 5000, 32, head_dim, device="cuda", dtype=torch.bfloat16)
    # skip all test
    attn = LiteAttention()
    attn.threshold = float('inf')
    # attn.threshold = float(0.0)
    torch.cuda.synchronize()
    output = attn(q, k, v)
    torch.cuda.synchronize()
    passed = (attn._skip_list[1, ..., 0] <= 2).all()
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("head_dim: ", head_dim)
    print("skip all test:", passed)
    if not passed:
        print(output.shape)
        print(attn._skip_list.shape)
        print(attn._skip_list[1, 0, 0, 0, :])

    # test must do
    attn = LiteAttention()
    attn.threshold = float('inf') # skip all
    torch.cuda.synchronize()
    must_do_list = [k.shape[1]-1,0]  # must not skip anything
    print("must_do_list", must_do_list)
    output = attn(q, k, v, must_do_list = must_do_list)
    torch.cuda.synchronize()
    passed = (attn._skip_list[1] == attn._skip_list[0]).all()  # check that nothing was skipped
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("head_dim: ", head_dim)
    print("skip all test:", passed)
    if not passed:
        print(output.shape)
        print(attn._skip_list.shape)
        print(attn._skip_list[1, 0, 0, 0, :])
    print("must do test:", passed)

    # skip nothing test
    attn = LiteAttention()
    attn.threshold = float('-inf')
    torch.cuda.synchronize()
    output = attn(q, k, v)
    torch.cuda.synchronize()
    passed = (attn._skip_list[1] == attn._skip_list[0]).all()
    print("skip nothing test:", passed)
    
    # Test softmax_lse correctness (with skip optimization disabled)
    attn = LiteAttention()
    attn.threshold = 0.0
    # # intentionally run twice to test how much the skip effects the lse
    # for i in range(2):
    for i in range(1):
        torch.cuda.synchronize()
        output_lite, lse_lite = attn(q, k, v, return_softmax_lse=True)
        torch.cuda.synchronize()
    
    # Compute reference softmax_lse using PyTorch
    # Shape: q, k, v are [batch, seq_len, num_heads, head_dim]
    scale = 1.0 / (head_dim ** 0.5)
    # Rearrange to [batch, num_heads, seq_len, head_dim] for bmm
    q_ref = q.transpose(1, 2).float()  # [batch, num_heads, seq_len, head_dim]
    k_ref = k.transpose(1, 2).float()  # [batch, num_heads, seq_len, head_dim]
    v_ref = v.transpose(1, 2).float()  # [batch, num_heads, seq_len, head_dim]
    
    # Compute attention scores: [batch, num_heads, seq_len, seq_len]
    scores = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * scale
    
    # Compute log-sum-exp along the last dimension (over keys)
    # logsumexp = log(sum(exp(scores), dim=-1))
    lse_ref = torch.logsumexp(scores, dim=-1)  # [batch, num_heads, seq_len]
    
    # Rearrange lse_lite to match reference shape
    # lse_lite shape is typically [batch, num_heads, seq_len]
    lse_lite_transposed = lse_lite
    if lse_lite.dim() == 4 and lse_lite.shape[-1] == 1:
        lse_lite_transposed = lse_lite.squeeze(-1)
    
    # Compare
    lse_diff = torch.abs(lse_ref - lse_lite_transposed.float())
    max_diff = lse_diff.max().item()
    mean_diff = lse_diff.mean().item()
    lse_passed = max_diff < 0.1  # Allow small numerical differences
    print(f"softmax_lse correctness test: {'PASSED' if lse_passed else 'FAILED'}")
    print(f"  max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f}")