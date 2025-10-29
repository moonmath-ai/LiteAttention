import torch
from lite_attention import LiteAttention


def generate_test_tensors(batch, seq_len, heads, head_dim):
    """Generate random Q, K, V tensors for testing."""
    q = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    return q, k, v


def run_attention_warmup(attn, q, k, v, num_iters=1):
    """Run attention forward pass multiple times to warm up."""
    for _ in range(num_iters):
        torch.cuda.synchronize()
        output = attn(q, k, v)
        torch.cuda.synchronize()
    return output

def print_skip_percentage(attn, q):
    """Print the skip percentage for the given query."""
    skip_percentage = attn.calc_percentage(attn._skip_list[attn._phase, :q.shape[0]])
    print(f"    Skip percentage: {skip_percentage:.2%}")

def test_skip_all(q, k, v, head_dim):
    """
    Test that when threshold is inf, all tiles are skipped except one range.
    Expected: skip_list should contain exactly 2 entries (one range of length 1).
    """
    attn = LiteAttention()
    attn.threshold = float('inf')
    
    # Warm up
    run_attention_warmup(attn, q, k, v)
    
    skip_list = attn._skip_list[attn._phase, :q.shape[0]]  # [batch, heads, qtiles, ktiles]
    
    # Test that skip lists include only 1 range (skip_list[..., 0] == 2 means 1 range)
    passed = (skip_list[..., 0] == 2).all()
    if not passed:
        print("  ⚠️  Skip list length is not 2")
    
    # Test that the only range has length 1
    diff = (skip_list[..., 1] - skip_list[..., 2]).abs()
    mpassed = (diff == 1)
    passed &= mpassed.all()
    
    print(f"  Skip all test: {'✅ PASSED' if passed else '❌ FAILED'}")
    if not passed:
        print(f"    Skip list shape: {skip_list.shape}")
        print_skip_percentage(attn, q)
        mdiff = diff[~mpassed]
        print(f"    Mismatched diffs: {mdiff}, shape: {mdiff.shape}")
        print(f"    Sample skip_list[0, 1, :, 1:3]:\n{skip_list[0, 1, :, 1:3]}")
    
    return passed


def test_skip_nothing(q, k, v, head_dim):
    """
    Test that when threshold is -inf, no tiles are skipped.
    Expected: skip lists should remain consistent between read and write phases.
    """
    attn = LiteAttention()
    attn.threshold = float('-inf')
    
    # Warm up
    run_attention_warmup(attn, q, k, v)
    
    read_list = attn._skip_list[attn._phase, :q.shape[0]]  # [batch, heads, qtiles, ktiles]
    write_list = attn._skip_list[1 - attn._phase, :q.shape[0]]  # [batch, heads, qtiles, ktiles]
    
    # Check if read and write lists match
    test_tensor = torch.tensor([2, read_list.shape[-1] - 2, -1], device=read_list.device, dtype=read_list.dtype)[None, None, None,]
    diff = (read_list[..., :3] == test_tensor).all(-1)
    passed = diff.all()
    
    print(f"  Skip nothing test: {'✅ PASSED' if passed else '❌ FAILED'}")
    if not passed:
        print_skip_percentage(attn, q)
        print(f"    Mismatched read_list:\n{read_list[~diff][..., :5]}")
    
    return passed


def compute_reference_lse(q, k, v, head_dim):
    """Compute reference softmax log-sum-exp using PyTorch."""
    scale = 1.0 / (head_dim ** 0.5)
    
    # Rearrange to [batch, num_heads, seq_len, head_dim]
    q_ref = q.transpose(1, 2).float()
    k_ref = k.transpose(1, 2).float()
    
    # Compute attention scores: [batch, num_heads, seq_len, seq_len]
    scores = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * scale
    
    # Compute log-sum-exp along the last dimension
    lse_ref = torch.logsumexp(scores, dim=-1)  # [batch, num_heads, seq_len]
    
    return lse_ref


def test_softmax_lse_correctness(q, k, v, head_dim, tolerance=0.001):
    """
    Test that softmax_lse output matches PyTorch reference implementation.
    """
    attn = LiteAttention()
    attn.threshold = 0.0
    
    torch.cuda.synchronize()
    output_lite, lse_lite = attn(q, k, v, return_softmax_lse=True)
    torch.cuda.synchronize()
    
    # Compute reference LSE
    lse_ref = compute_reference_lse(q, k, v, head_dim)
    
    # Adjust lse_lite shape if needed
    lse_lite_transposed = lse_lite
    if lse_lite.dim() == 4 and lse_lite.shape[-1] == 1:
        lse_lite_transposed = lse_lite.squeeze(-1)
    
    # Compare
    lse_diff = torch.abs(lse_ref - lse_lite_transposed.float())
    max_diff = lse_diff.max().item()
    mean_diff = lse_diff.mean().item()
    passed = max_diff < tolerance
    
    print(f"  Softmax LSE test: {'✅ PASSED' if passed else '❌ FAILED'}")
    print(f"    Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
    
    return passed

def stress_test(q, k, v, head_dim, num_iters=10):
    """Stress test the attention mechanism."""
    attn = LiteAttention()
    attn.threshold = float(0.0)

    torch.cuda.synchronize()
    output = attn(q, k, v)
    torch.cuda.synchronize()

    n = 11
    percentage = attn.calc_percentage(attn._skip_list[attn._phase, :q.shape[0]])
    print(f"  Skip list: {attn._skip_list[attn._phase, 0,0,0,:n]}, ktiles: {attn._skip_list.shape[-1] - 1}")

    for i in range(num_iters):
        torch.cuda.synchronize()
        output = attn(q, k, v)
        torch.cuda.synchronize()
        new_percentage = attn.calc_percentage(attn._skip_list[attn._phase, :q.shape[0]])
        if new_percentage != percentage:
            print(f"  Skip list: {attn._skip_list[attn._phase, 0,0,0,:n]}, ktiles: {attn._skip_list.shape[-1] - 1}")
            print(f"  percentage changed from {percentage:.2%} to {new_percentage:.2%} at iteration {i}")
            print(f"  Stress test completed: {'✅ PASSED' if False else '❌ FAILED'}")
            return

    print_skip_percentage(attn, q)
    print(f"  Stress test completed: {'✅ PASSED' if True else '❌ FAILED'}")
    

def run_tests_for_head_dim(head_dim, batch=2, seq_len=18200, heads=32):
    """Run all tests for a specific head dimension."""
    print(f"\n{'='*60}")
    print(f"Testing head_dim: {head_dim}")
    print(f"{'='*60}")
    
    # Generate test data
    q, k, v = generate_test_tensors(batch, seq_len, heads, head_dim)
    
    # Run tests
    stress_test(q, k, v, head_dim)
    test_skip_all(q, k, v, head_dim)
    test_skip_nothing(q, k, v, head_dim)

    q, k, v = generate_test_tensors(batch=batch, seq_len=min(6000, seq_len), heads=heads, head_dim=head_dim)
    test_softmax_lse_correctness(q, k, v, head_dim)


def main():
    """Main test runner."""
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    # Test different head dimensions
    head_dims = [32, 64, 96, 128, 192, 256]
    
    for head_dim in head_dims:
        run_tests_for_head_dim(head_dim)
    
    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()