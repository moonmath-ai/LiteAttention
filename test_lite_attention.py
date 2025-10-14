import torch
from lite_attention import LiteAttention

torch.manual_seed(0)
torch.cuda.manual_seed(0)

for head_dim in [32, 64, 96, 128, 192, 256]:
    print(f"head_dim: {head_dim}")
    q = torch.randn(2, 10000, 32, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 10000, 32, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 10000, 32, head_dim, device="cuda", dtype=torch.bfloat16)
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

    # skip nothing test
    attn = LiteAttention()
    attn.threshold = float('-inf')
    torch.cuda.synchronize()
    output = attn(q, k, v)
    torch.cuda.synchronize()
    passed = (attn._skip_list[1] == attn._skip_list[1]).all()
    print("skip nothing test:", passed)