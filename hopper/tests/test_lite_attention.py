import pytest
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
    skip_percentage = attn.calc_percentage(attn.read_list)
    print(f"    Skip percentage: {skip_percentage:.2%}", f"raw percentage: {skip_percentage}")

# not valid in the new skip list format!!!
def check_first_element_is_last_block(skip_list):
    """
    Check that the first element in the skip list is the last block (ktiles - 1).
    
    Args:
        skip_list: Skip list tensor of shape [batch, heads, qtiles, ktiles]
    
    Returns:
        bool: True if all first elements equal the last block index, False otherwise.
    """
    last_n_block = skip_list.shape[-1] - 2
    is_n_block = skip_list[..., 1] == last_n_block
    is_all_n_blocks = is_n_block.all()
    if not is_all_n_blocks:
        print(f"     ⚠️  First Element is not ktiles - 1!, it's: {skip_list[..., 1]} != {last_n_block}")
    return is_all_n_blocks

def check_skip_list_length_valid(skip_list):
    """
    Check that the list length isn't bigger than ktiles + 1.
    
    Args:
        skip_list: Skip list tensor of shape [batch, heads, qtiles, ktiles]
    
    Returns:
        bool: True if all list lengths are valid, False otherwise.
    """
    passed = (skip_list.shape[-1] > skip_list[..., 0]).all()
    if not passed:
        print(f"      ⚠️  List length is bigger than the length of the skip list: {skip_list[..., 0]} <= {skip_list.shape[-1]}")
    return passed

def check_no_empty_or_negative_ranges(skip_list):
    """
    Check that we don't have empty or negative ranges in the skip list.
    
    Args:
        skip_list: Skip list tensor of shape [batch, heads, qtiles, ktiles]
    
    Returns:
        bool: True if no empty or negative ranges exist, False otherwise.
    """
    # Check that all ranges are positive (start < end)
    # [start0 - end0, end0 - start1, start1 - end1, end1 - start2, ..., start_n - end_n]
    diff = (skip_list[..., 1:-1] - skip_list[..., 2:])
    # correct the sign according to the first difference
    sign = torch.sign(diff.flatten()[0])
    diff = (diff * sign) > 0

    arange = torch.arange(diff.shape[-1], device=skip_list.device).view(1, 1, 1, -1) >= skip_list[..., 0:1] - 1
    # Only check ranges that are within the valid list length
    passed_individually = (arange + diff) > 0
    passed_individually = passed_individually.all(-1)
    passed = passed_individually.all()
    if not passed:
        print(f"     ⚠️  Empty or negative ranges found!")
        not_passed = skip_list[~passed_individually]
        max_len = (not_passed[..., 0].flatten().max() + 1).item()
        print(f"    Failed items: {not_passed[..., :max_len]}")
    return passed

def _test_skip_all(q, k, v, head_dim):
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
    
    # Test that the only range has length 1
    diff = (skip_list[..., 1] - skip_list[..., 2]).abs()
    passed &= (diff == 1).all()
    
    print(f"  Skip all test: {'✅ PASSED' if passed else '❌ FAILED'}")
    if not passed:
        print(f"    Skip list shape: {skip_list.shape}")
        print_skip_percentage(attn, q)
    
    assert passed, "Skip all test failed"


def _test_skip_nothing(q, k, v, head_dim):
    """
    Test that when threshold is -inf, no tiles are skipped.
    Expected: skip lists should remain consistent between read and write phases.
    """
    attn = LiteAttention()
    attn.threshold = float('-inf')
    read_list_original, _ = attn._get_read_write_lists(q, v)
    read_list_original = read_list_original.clone()
    attn._phase = 0
    
    # Warm up
    run_attention_warmup(attn, q, k, v, 2)
    
    # read_list = attn._skip_list[attn._phase, :q.shape[0]]  # [batch, heads, qtiles, ktiles]
    read_list = attn.read_list  # [batch, heads, qtiles, ktiles+1]
    # write_list = attn._skip_list[1 - attn._phase, :q.shape[0]]  # [batch, heads, qtiles, ktiles]
    # write_list = attn.write_list  # [batch, heads, qtiles, ktiles+1]
    
    # Check if read and write lists match
    one_range = read_list[..., 0] == 2
    diff_min = read_list[..., 1 : 3].min(dim=-1).values == read_list_original[..., 1 : 3].min(dim=-1).values
    diff_max = read_list[..., 1 : 3].max(dim=-1).values == read_list_original[..., 1 : 3].max(dim=-1).values
    assert diff_min.shape == diff_max.shape == one_range.shape
    diff = one_range & diff_min & diff_max
    passed = diff.all()
    mismatch_percent = (~diff).sum().item() / diff.numel() * 100
    
    print(f"  Skip nothing test: {'✅ PASSED' if passed else '❌ FAILED'}")
    if not passed:
        print(f"    Mismatch percentage: {mismatch_percent:.2f}%")
        print_skip_percentage(attn, q)
    
    assert passed, f"Read list mismatch {mismatch_percent:.2f}%"


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


def _test_softmax_lse_correctness(q, k, v, head_dim, tolerance=0.001):
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
    
    assert passed, f"Max diff {max_diff:.6f} exceeds tolerance {tolerance}"

def _test_consistency(q, k, v, head_dim, num_iters=10):
    """Test that the skip list is consistent between reads and writes."""
    attn = LiteAttention()
    attn.threshold = float(0.0)

    previous_skip_list = None
    skip_list = None
    percentage = float('inf')
    passed = True
    
    for i in range(num_iters):
        q, k, v = generate_test_tensors(batch=q.shape[0], seq_len=q.shape[1], heads=q.shape[2], head_dim=q.shape[3])
        torch.cuda.synchronize()
        output = attn(q, k, v)
        torch.cuda.synchronize()

        previous_skip_list = skip_list
        # skip_list = attn._skip_list[attn._phase, :q.shape[0]]
        skip_list = attn.read_list

        # check new percentage is not bigger than the previous one
        new_percentage = attn.calc_percentage(skip_list)
        if new_percentage > percentage:
            print(f"  Consistency test: ❌ FAILED")
            print(f"    Failed on iteration {i}")
            print(f"    New percentage is bigger than the previous one: {new_percentage:.2%} > {percentage:.2%}")
            passed = False
            break
        percentage = new_percentage
        
        # Check that the list length isn't bigger than ktiles + 1
        if not check_skip_list_length_valid(skip_list):
            print(f"  Consistency test: ❌ FAILED")
            print(f"    Failed on iteration {i}")
            passed = False
            break

        # Check that we don't have empty or negative ranges
        if not check_no_empty_or_negative_ranges(skip_list):
            print(f"  Consistency test: ❌ FAILED")
            print(f"    Failed on iteration {i}")
            passed = False
            break
    
    if passed:
        print(f"  Consistency test: ✅ PASSED")
    
    assert passed, "Consistency test failed"

def get_must_skip_list_cases(seq_len):
    """Generate must skip list test cases based on sequence length."""
    return [
        ("beginning_and_end", [0, 1000, 10000, seq_len-1]),
        ("first_half", [0, 5000]),
        ("middle_quarter", [seq_len // 4, seq_len // 2]),
        ("first_and_last_10pct", [0, seq_len // 10, seq_len * 9 // 10, seq_len-1]),
        ("middle_third", [seq_len // 3, seq_len * 2 // 3]),
        ("multiple_small_ranges", [0, 2000, 5000, 7000, 10000, seq_len-1]),
    ]


def _test_must_skip_list_single(q, k, v, head_dim, must_skip_list, case_name):
    """
    Test that must_skip_list forces tiles to be skipped even if threshold dictates computing.
    Tests a single must skip list configuration.
    """
    seq_len = k.shape[1]
    element_size = k.dtype.itemsize
    _, kBlockN = LiteAttention.get_MN(head_dim, element_size)
    ktiles = LiteAttention.ceil_div(seq_len, kBlockN)

    attn = LiteAttention()
    # Set threshold to -inf to compute everything by default
    attn.threshold = -float("inf")

    torch.cuda.synchronize()
    output = attn(q, k, v, must_skip_list=must_skip_list)
    torch.cuda.synchronize()
    
    # The write_list from this pass (which will be read_list next pass)
    # should contain the skip information.
    result_list = attn.read_list
    
    # Calculate expected percentage based on tiles
    skipped_tiles = 0
    for i in range(0, len(must_skip_list), 2):
        start_seq = must_skip_list[i]
        end_seq = must_skip_list[i+1]
        start_tile = start_seq // kBlockN
        end_tile = LiteAttention.ceil_div(end_seq, kBlockN)
        skipped_tiles += end_tile - start_tile
    expected_percentage = (ktiles - skipped_tiles) / ktiles
    
    actual_percentage = attn.calc_percentage(result_list)
    passed = abs(actual_percentage - expected_percentage) < 0.01
    
    print(f"    {case_name}: {'✅ PASSED' if passed else '❌ FAILED'}")
    if not passed:
        print(f"      Expected {expected_percentage:.2%} computed, got {actual_percentage:.2%}")
    
    assert passed, f"Case '{case_name}': Expected {expected_percentage:.2%}, got {actual_percentage:.2%}"

def get_must_do_list_cases(seq_len):
    """Generate must do list test cases based on sequence length."""
    return [
        ("beginning_and_end", [0, 1000, 10000, seq_len-1]),
        ("first_half", [0, 5000]),
        ("middle_quarter", [seq_len // 4, seq_len // 2]),
        ("first_and_last_10pct", [0, seq_len // 10, seq_len * 9 // 10, seq_len-1]),
        ("middle_third", [seq_len // 3, seq_len * 2 // 3]),
        ("multiple_small_ranges", [0, 2000, 5000, 7000, 10000, seq_len-1]),
        ("custom_test", [0, 2000, 15000, seq_len-1]),
    ]


def _test_must_do_list_single(q, k, v, head_dim, must_do_list, case_name, num_iters=10):
    """
    Test that must_do_list forces tiles to be computed even if threshold dictates skipping.
    Tests a single must do list configuration.
    """
    seq_len = k.shape[1]
    element_size = k.dtype.itemsize
    _, kBlockN = LiteAttention.get_MN(head_dim, element_size)
    ktiles = LiteAttention.ceil_div(seq_len, kBlockN)

    attn = LiteAttention()
    # Set threshold to +inf to skip everything by default
    attn.threshold = float("inf")

    passed = True
    for i in range(num_iters):
        torch.cuda.synchronize()
        output = attn(q, k, v, must_do_list=must_do_list)
        torch.cuda.synchronize()
        
        # The write_list from this pass (which will be read_list next pass)
        # should contain the compute information.
        result_list = attn.read_list
        
        # Calculate expected percentage based on tiles
        computed_tiles = 0
        for j in range(0, len(must_do_list), 2):
            start_seq = must_do_list[j]
            end_seq = must_do_list[j+1]
            start_tile = start_seq // kBlockN
            end_tile = LiteAttention.ceil_div(end_seq, kBlockN)
            computed_tiles += end_tile - start_tile
        expected_percentage = computed_tiles / ktiles
        
        actual_percentage = attn.calc_percentage(result_list)
        if abs(actual_percentage - expected_percentage) >= 0.01:
            passed = False
            print(f"    {case_name}: ❌ FAILED (iter {i})")
            print(f"      Expected {expected_percentage:.2%} computed, got {actual_percentage:.2%}")
            break
    
    if passed:
        print(f"    {case_name}: ✅ PASSED")
    
    assert passed, f"Case '{case_name}': Test failed"

def _test_stress(q, k, v, head_dim, num_iters=10):
    """Stress test the attention mechanism."""
    attn = LiteAttention()
    attn.threshold = float(0.0)

    output = run_attention_warmup(attn, q, k, v, 2) # only after 2 iters we stabalize do to bi-direction

    n = 11
    percentage = attn.calc_percentage(attn.read_list)
    read_list_original = attn.read_list.clone()
    percentage_per_head = attn.calc_percentage_per_head(attn.read_list)
    
    passed = True

    for i in range(num_iters):
        torch.cuda.synchronize()
        output = attn(q, k, v)
        torch.cuda.synchronize()
        new_percentage = attn.calc_percentage(attn.read_list)
        new_percentage_per_head = attn.calc_percentage_per_head(attn.read_list)
        
        if new_percentage != percentage:
            print(f"  Skip list: {attn._skip_list[attn._phase, 0,0,0,:n]}, ktiles: {attn._skip_list.shape[-1] - 1}")
            print(f"  percentage changed from {percentage} to {new_percentage} at iteration {i}")
            print(f"  Stress test: ❌ FAILED")
            diff = new_percentage_per_head != percentage_per_head
            diff_read = attn.read_list[diff]
            diff_read_original = read_list_original[diff]
            length = max(diff_read[..., 0].max().item(), diff_read_original[..., 0].max().item())
            print(f"  read_list: {diff_read[..., :length]}")
            print(f"  original : {diff_read_original[..., :length]}")
            passed = False
            break
        
        percentage = new_percentage

    if passed:
        print_skip_percentage(attn, q)
        print(f"  Stress test: ✅ PASSED")
    
    assert passed, f"Percentage changed unexpectedly"
    

# Pytest fixtures and parametrized tests
@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducibility before each test."""
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)


# @pytest.fixture(params=[32, 64, 96, 128, 192, 256])
# TODO: check small head dims after the fix
@pytest.fixture(params=[128, 192, 256])
def head_dim(request):
    """Parametrize tests across different head dimensions."""
    return request.param


@pytest.fixture
def test_tensors(head_dim):
    """Generate test tensors for a given head dimension."""
    batch, seq_len, heads = 2, 18200, 32
    return generate_test_tensors(batch, seq_len, heads, head_dim)


@pytest.fixture
def small_test_tensors(head_dim):
    """Generate smaller test tensors for LSE tests."""
    batch, seq_len, heads = 2, 6000, 32
    return generate_test_tensors(batch, seq_len, heads, head_dim)


# Parametrized test wrappers that use fixtures
def test_stress_parametrized(test_tensors, head_dim):
    """Stress test with parametrized head dimensions."""
    q, k, v = test_tensors
    _test_stress(q, k, v, head_dim)


def test_skip_all_parametrized(test_tensors, head_dim):
    """Test skip all with parametrized head dimensions."""
    q, k, v = test_tensors
    _test_skip_all(q, k, v, head_dim)


def test_skip_nothing_parametrized(test_tensors, head_dim):
    """Test skip nothing with parametrized head dimensions."""
    q, k, v = test_tensors
    _test_skip_nothing(q, k, v, head_dim)


@pytest.mark.parametrize("case_idx", [0, 1, 2, 3, 4, 5], ids=[
    "beginning_and_end",
    "first_half", 
    "middle_quarter",
    "first_and_last_10pct",
    "middle_third",
    "multiple_small_ranges"
])
def test_must_skip_list_parametrized(test_tensors, head_dim, case_idx):
    """Test must skip list with parametrized head dimensions and test cases."""
    if case_idx == 0:
        print("  Must-skip list tests:")
    q, k, v = test_tensors
    seq_len = k.shape[1]
    cases = get_must_skip_list_cases(seq_len)
    case_name, must_skip_list = cases[case_idx]
    _test_must_skip_list_single(q, k, v, head_dim, must_skip_list, case_name)


@pytest.mark.parametrize("case_idx", [0, 1, 2, 3, 4, 5, 6], ids=[
    "beginning_and_end",
    "first_half",
    "middle_quarter", 
    "first_and_last_10pct",
    "middle_third",
    "multiple_small_ranges",
    "custom_test"
])
def test_must_do_list_parametrized(test_tensors, head_dim, case_idx):
    """Test must do list with parametrized head dimensions and test cases."""
    if case_idx == 0:
        print("  Must-do list tests:")
    q, k, v = test_tensors
    seq_len = k.shape[1]
    cases = get_must_do_list_cases(seq_len)
    case_name, must_do_list = cases[case_idx]
    _test_must_do_list_single(q, k, v, head_dim, must_do_list, case_name)


def test_softmax_lse_correctness_parametrized(small_test_tensors, head_dim):
    """Test softmax LSE correctness with parametrized head dimensions."""
    q, k, v = small_test_tensors
    _test_softmax_lse_correctness(q, k, v, head_dim)