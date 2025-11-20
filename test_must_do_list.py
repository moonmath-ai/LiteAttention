import torch
from lite_attention import LiteAttention

def generate_test_tensors(batch, seq_len, heads, head_dim):
    """Generate random Q, K, V tensors for testing."""
    q = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    return q, k, v

def test_must_do_list(q, k, v, head_dim):
    """
    Test that must_do_list forces tiles to be computed even if threshold dictates skipping.
    """
    attn = LiteAttention()
    # Set threshold to infinity to skip almost everything by default
    # Note: Direct assignment bypasses validation (set_threshold requires negative values)
    # This is intentional for testing extreme skipping behavior
    attn.threshold = float("inf")
    
    # Define a must-do range: sequence positions [32, 64) (32 tokens)
    # This ensures that even with infinite threshold, these tiles should be computed
    must_do_list = [0, k.shape[1]]
    
    torch.cuda.synchronize()
    output = attn(q, k, v, must_do_list=must_do_list)
    torch.cuda.synchronize()
    
    # The write_list from this pass (which will be read_list next pass)
    # should contain the forced tiles.
    # We access it via read_list property because the phase has switched.
    result_list = attn.read_list
    
    # Check if result_list is not empty (all skipped)
    # If threshold=inf and no must_do, percentage should be ~0 (except maybe last block)
    # With must_do, it should be significantly > 0 if seq_len is small, or just > 0.
    
    percentage = attn.calc_percentage(result_list)
    passed = percentage == 1.0
    
    print(f"  Must-do list test: {'✅ PASSED' if passed else '❌ FAILED'}")
    if not passed:
        print(f"    Expected == 0% computed, got {percentage:.2%}")
        
    return passed

def run_tests_for_head_dim(head_dim, batch=2, seq_len=18200, heads=32):
    """Run all tests for a specific head dimension."""
    print(f"\n{'='*60}")
    print(f"Testing head_dim: {head_dim}")
    print(f"{'='*60}")
    
    # Generate test data
    q, k, v = generate_test_tensors(batch, seq_len, heads, head_dim)
    
    # Run tests
    test_must_do_list(q, k, v, head_dim)


def main():
    """Main test runner."""
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    # Test different head dimensions
    head_dims = [128, 192, 256]
    
    for head_dim in head_dims:
        run_tests_for_head_dim(head_dim)
    
    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()