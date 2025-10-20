import torch
from lite_attention import LiteAttention

torch.manual_seed(0)
torch.cuda.manual_seed(0)

def init_qkv(head_dim):
    q = torch.randn(1, 2*5000, 4, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2*5000, 4, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 2*5000, 4, head_dim, device="cuda", dtype=torch.bfloat16)
    return q, k, v

for head_dim in [32, 64, 96, 128, 192, 256]:
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"head_dim: {head_dim}")
    attn = LiteAttention()
    attn.threshold = float(2)
    for i in range(2):
        q, k, v = init_qkv(head_dim)
        output = attn(q, k, v)
    torch.cuda.synchronize()
    print(attn._skip_list.shape)