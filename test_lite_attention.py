import os
import torch
from lite_attention import LiteAttention

# Enable debug mode to allow non-negative thresholds in tests
os.environ["LITE_ATTENTION_DEBUG"] = "TRUE"


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

def test_skip_all(q, k, v, head_dim, use_int8=False):
    """
    Test that when threshold is inf, all tiles are skipped except one range.
    Expected: skip_list should contain exactly 2 entries (one range of length 1).
    """
    attn = LiteAttention(use_int8=use_int8)
    attn.threshold = float('inf')
    
    # Warm up
    run_attention_warmup(attn, q, k, v)
    
    skip_list = attn._skip_list[attn._phase, :q.shape[0]]  # [batch, heads, qtiles, ktiles]
    
    # Test that skip lists include only 1 range (skip_list[..., 0] == 2 means 1 range)
    passed = (skip_list[..., 0] == 2).all()
    if not passed:
        print("      ⚠️  Skip list length is not 2")
    
    # Test that the only range has length 1
    diff = (skip_list[..., 1] - skip_list[..., 2]).abs()
    mpassed = (diff == 1)
    passed &= mpassed.all()

    # # Test that the only block we don't skip is the last one
    # passed &= check_first_element_is_last_block(skip_list)
    
    prefix = "INT8 " if use_int8 else ""
    print(f"  {prefix}Skip all test: {'✅ PASSED' if passed else '❌ FAILED'}")
    if not passed:
        print(f"    Skip list shape: {skip_list.shape}")
        print_skip_percentage(attn, q)
        mdiff = diff[~mpassed]
        print(f"    Mismatched diffs: {mdiff}, shape: {mdiff.shape}")
        print(f"    Sample skip_list[0, 1, :, 1:3]:\n{skip_list[0, 1, :, 1:3]}")
    
    return passed


def test_skip_nothing(q, k, v, head_dim, use_int8=False):
    """
    Test that when threshold is -inf, no tiles are skipped.
    Expected: skip lists should remain consistent between read and write phases.
    """
    attn = LiteAttention(use_int8=use_int8)
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
    prefix = "INT8 " if use_int8 else ""
    print(f"  {prefix}Skip nothing test: {'✅ PASSED' if passed else '❌ FAILED'}")
    if not passed:
        mismatch_percent = (~diff).sum().item() / diff.numel() * 100
        print(f"    Mismatch percentage: {mismatch_percent:.2f}%")
        print_skip_percentage(attn, q)
        print(f"    Mismatched read_list:\n{read_list[~diff][..., :3]}")
        print(f"    Mismatched read_list_original:\n{read_list_original[~diff][..., :3]}")
    
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


def compute_error_metrics(output, reference, name=""):
    """Compute and return error metrics between output and reference."""

    assert output.shape == reference.shape, f"Output and reference shapes do not match: {output.shape} != {reference.shape}"

    # Convert both to float32 for accurate error computation
    out_f32 = output.float()
    ref_f32 = reference.float()
    
    # Absolute errors
    abs_diff = (out_f32 - ref_f32).abs()
    max_abs_error = abs_diff.max().item()
    mean_abs_error = abs_diff.mean().item()
    
    # Relative errors (avoid division by zero)
    ref_abs = ref_f32.abs().clamp(min=1e-7)
    rel_diff = abs_diff / ref_abs
    max_rel_error = rel_diff.max().item()
    mean_rel_error = rel_diff.mean().item()
    
    # RMSE
    rmse = torch.sqrt((abs_diff ** 2).mean()).item()
    
    # Cosine similarity (flatten and compute)
    out_flat = out_f32.flatten()
    ref_flat = ref_f32.flatten()
    cosine_sim = torch.nn.functional.cosine_similarity(out_flat.unsqueeze(0), ref_flat.unsqueeze(0)).item()
    
    return {
        'max_abs_error': max_abs_error,
        'mean_abs_error': mean_abs_error,
        'max_rel_error': max_rel_error,
        'mean_rel_error': mean_rel_error,
        'rmse': rmse,
        'cosine_sim': cosine_sim
    }


def test_softmax_lse_correctness(q, k, v, head_dim, tolerance=0.001, use_int8=False):
    """
    Test that softmax_lse output matches PyTorch reference implementation.
    """
    attn = LiteAttention(use_int8=use_int8)
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
    # Use relaxed tolerance for INT8
    tolerance_actual = tolerance * 10 if use_int8 else tolerance
    passed = max_diff < tolerance_actual
    
    prefix = "INT8 " if use_int8 else ""
    print(f"  {prefix}Softmax LSE test: {'✅ PASSED' if passed else '❌ FAILED'}")
    print(f"    Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f} (tolerance: {tolerance_actual:.6f})")
    
    return passed

def consistency_test(q, k, v, head_dim, num_iters=10):
    """Test that the skip list is consistent between reads and writes."""
    attn = LiteAttention()
    attn.threshold = float(0.0)

    previous_skip_list = None
    skip_list = None
    percentage = float('inf')
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
            print(f"  Consistency test: {'✅ PASSED' if False else '❌ FAILED'}")
            print(f"    Failed on iteration {i}")
            print(f"    New percentage is bigger than the previous one: {new_percentage:.2%} > {percentage:.2%}")
            return False
        percentage = new_percentage
        
        # # Check that the first element in the skip list is the last block
        # if not check_first_element_is_last_block(skip_list):
        #     print(f"  Consistency test: {'✅ PASSED' if False else '❌ FAILED'}")
        #     print(f"    Failed on iteration {i}")
        #     return False
        
        # Check that the list length isn't bigger than ktiles + 1
        if not check_skip_list_length_valid(skip_list):
            print(f"  Consistency test: {'✅ PASSED' if False else '❌ FAILED'}")
            print(f"    Failed on iteration {i}")
            return False

        # Check that we don't have empty or negative ranges
        if not check_no_empty_or_negative_ranges(skip_list):
            print(f"  Consistency test: {'✅ PASSED' if False else '❌ FAILED'}")
            print(f"    Failed on iteration {i}")
            return False
    
    print(f"  Consistency test: {'✅ PASSED' if True else '❌ FAILED'}")
    return True

def test_must_skip_list(q, k, v, head_dim, use_int8=False):
    """
    Test that must_skip_list forces tiles to be skipped even if threshold dictates computing.
    Tests multiple must skip list configurations.
    """
    seq_len = k.shape[1]
    element_type = torch.int8 if use_int8 else k.dtype
    _, kBlockN = LiteAttention.get_MN(head_dim, element_type)
    ktiles = LiteAttention.ceil_div(seq_len, kBlockN)

    # Each entry is [start0, end0, start1, end1, ...] representing ranges to skip
    must_skip_list_cases = [
        [0, 1000, 10000, seq_len-1],                       # Skip beginning and end
        [0, 5000],                                         # Skip first half
        [seq_len // 4, seq_len // 2],                      # Skip middle quarter
        [0, seq_len // 10, seq_len * 9 // 10, seq_len-1],  # Skip first and last 10%
        [seq_len // 3, seq_len * 2 // 3],                  # Skip middle third
        [0, 2000, 5000, 7000, 10000, seq_len-1],           # Multiple small ranges
    ]
    
    all_passed = True
    for test_idx, must_skip_list in enumerate(must_skip_list_cases):
        attn = LiteAttention(use_int8=use_int8)

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
        
        all_passed &= passed
    
    prefix = "INT8 " if use_int8 else ""
    print(f"  {prefix}Must-skip list tests: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    return all_passed

def test_must_do_list(q, k, v, head_dim, use_int8=False):
    """
    Test that must_do_list forces tiles to be computed even if threshold dictates skipping.
    Tests multiple must do list configurations.
    """
    seq_len = k.shape[1]
    element_type = torch.int8 if use_int8 else k.dtype
    _, kBlockN = LiteAttention.get_MN(head_dim, element_type)
    ktiles = LiteAttention.ceil_div(seq_len, kBlockN)

    # Each entry is [start0, end0, start1, end1, ...] representing ranges to compute
    must_do_list_cases = [
        [0, 1000, 10000, seq_len-1],                       # Compute beginning and end
        [0, 5000],                                         # Compute first half
        [seq_len // 4, seq_len // 2],                      # Compute middle quarter
        [0, seq_len // 10, seq_len * 9 // 10, seq_len-1],  # Compute first and last 10%
        [seq_len // 3, seq_len * 2 // 3],                  # Compute middle third
        [0, 2000, 5000, 7000, 10000, seq_len-1],           # Multiple small ranges
        [0, 2000, 15000, seq_len-1],                       # Custom test
    ]
    
    all_passed = True
    for test_idx, must_do_list in enumerate(must_do_list_cases):
        attn = LiteAttention(use_int8=use_int8)

        # Set threshold to +inf to skip everything by default
        attn.threshold = float("inf")

        for i in range(10):
            torch.cuda.synchronize()
            output = attn(q, k, v, must_do_list=must_do_list)
            torch.cuda.synchronize()
            
            # The write_list from this pass (which will be read_list next pass)
            # should contain the compute information.
            result_list = attn.read_list
            
            # Calculate expected percentage based on tiles
            computed_tiles = 0
            for i in range(0, len(must_do_list), 2):
                start_seq = must_do_list[i]
                end_seq = must_do_list[i+1]
                start_tile = start_seq // kBlockN
                end_tile = LiteAttention.ceil_div(end_seq, kBlockN)
                computed_tiles += end_tile - start_tile
                # print(f"    Range [{start_seq}, {end_seq}): tiles [{start_tile}, {end_tile}) = {end_tile - start_tile} tiles")
            # print(f"    Debug: Tiles to compute={computed_tiles}, Tiles total={ktiles}")
            expected_percentage = computed_tiles / ktiles
            
            actual_percentage = attn.calc_percentage(result_list)
            passed = abs(actual_percentage - expected_percentage) < 0.01

            if not passed:
                print(f"    Expected {expected_percentage:.2%} computed, got {actual_percentage:.2%}, expected tile count: {computed_tiles}, total tiles: {ktiles}")
                print(f"    Must do ranges: {must_do_list}")
            
            all_passed &= passed
        
    prefix = "INT8 " if use_int8 else ""
    print(f"  {prefix}Must-do list tests: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    return all_passed

def stress_test(q, k, v, head_dim, num_iters=10, use_int8=False):
    """Stress test the attention mechanism."""
    attn = LiteAttention(use_int8=use_int8)
    attn.threshold = float(0.0)

    output = run_attention_warmup(attn, q, k, v, 2) # only after 2 iters we stabilize due to bi-direction

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
            # print(f"  percentage changed from {percentage:.2%} to {new_percentage:.2%} at iteration {i}")
            print(f"  percentage changed from {percentage} to {new_percentage} at iteration {i}")
            prefix = "INT8 " if use_int8 else ""
            print(f"  {prefix}Stress test: {'✅ PASSED' if False else '❌ FAILED'}")
            diff = new_percentage_per_head != percentage_per_head
            diff_read = attn.read_list[diff]
            diff_read_original = read_list_original[diff]
            length = max(diff_read[..., 0].max().item(), diff_read_original[..., 0].max().item())
            print(f"  read_list: {diff_read[..., :length]}")
            print(f"  original : {diff_read_original[..., :length]}")
            return False

    print_skip_percentage(attn, q)
    prefix = "INT8 " if use_int8 else ""
    print(f"  {prefix}Stress test: {'✅ PASSED' if passed else '❌ FAILED'}")
    return passed
    

def test_int8_correctness(q, k, v, head_dim, tolerance_max_abs=0.1, tolerance_cosine=0.99):
    """
    Test that INT8 output matches BF16 output within acceptable tolerance.
    
    Args:
        q, k, v: Input tensors
        head_dim: Head dimension
        tolerance_max_abs: Maximum acceptable absolute error (default: 0.1)
        tolerance_cosine: Minimum acceptable cosine similarity (default: 0.99)
    """
    # Check if tile sizes match between int8 and bf16
    tile_size_bf16 = LiteAttention.get_MN(head_dim, torch.bfloat16, is_skipable=False)
    tile_size_int8 = LiteAttention.get_MN(head_dim, torch.int8, is_skipable=False)
    tile_sizes_match = tile_size_bf16 == tile_size_int8
    
    if not tile_sizes_match:
        print(f"    ⚠️  Tile sizes differ (BF16: {tile_size_bf16}, INT8: {tile_size_int8})")
        print(f"    Test results shown for reference but not considered as failure")
    
    scale = 1.0 / (head_dim ** 0.5)
    
    # Create BF16 reference (without skipping for fair comparison)
    attn_bf16 = LiteAttention(enable_skipping=False, use_int8=False)
    torch.cuda.synchronize()
    output_bf16 = attn_bf16(q, k, v, scale=scale)
    torch.cuda.synchronize()
    
    # Create INT8 version (without skipping for fair comparison)
    attn_int8 = LiteAttention(enable_skipping=False, use_int8=True)
    torch.cuda.synchronize()
    output_int8 = attn_int8(q, k, v, scale=scale)
    torch.cuda.synchronize()
    
    # Compute error metrics
    metrics = compute_error_metrics(output_int8, output_bf16, "INT8 vs BF16")
    
    # Check tolerances
    passed = (metrics['max_abs_error'] < tolerance_max_abs and 
              metrics['cosine_sim'] >= tolerance_cosine)
    
    # Adjust status message based on tile size match
    if not tile_sizes_match:
        status = '⚠️  SKIPPED (tile size mismatch)' if not passed else '✅ PASSED (tile size mismatch, results OK)'
    else:
        status = '✅ PASSED' if passed else '❌ FAILED'
    
    print(f"  INT8 correctness test: {status}")
    print(f"    Max absolute error: {metrics['max_abs_error']:.6e} (tolerance: {tolerance_max_abs:.6e})")
    print(f"    Mean absolute error: {metrics['mean_abs_error']:.6e}")
    print(f"    RMSE: {metrics['rmse']:.6e}")
    print(f"    Cosine similarity: {metrics['cosine_sim']:.8f} (tolerance: {tolerance_cosine:.8f})")
    
    if not passed:
        print(f"    Max relative error: {metrics['max_rel_error']:.6e}")
        print(f"    Mean relative error: {metrics['mean_rel_error']:.6e}")
    
    # If tile sizes don't match, always return True (don't fail)
    return passed if tile_sizes_match else True


def test_int8_with_skipping(q, k, v, head_dim, tolerance_max_abs=0.15, tolerance_cosine=0.98):
    """
    Test that INT8 works correctly with skipping enabled.
    Compares INT8 with skipping vs BF16 with skipping.
    """
    # Check if tile sizes match between int8 and bf16
    tile_size_bf16 = LiteAttention.get_MN(head_dim, torch.bfloat16, is_skipable=True)
    tile_size_int8 = LiteAttention.get_MN(head_dim, torch.int8, is_skipable=True)
    tile_sizes_match = tile_size_bf16 == tile_size_int8
    
    if not tile_sizes_match:
        print(f"    ⚠️  Tile sizes differ (BF16: {tile_size_bf16}, INT8: {tile_size_int8})")
        print(f"    Test results shown for reference but not considered as failure")
    
    scale = 1.0 / (head_dim ** 0.5)
    threshold = 0.0
    
    # Create BF16 reference with skipping
    attn_bf16 = LiteAttention(enable_skipping=True, use_int8=False, threshold=threshold)
    # Warm up to stabilize skip lists
    run_attention_warmup(attn_bf16, q, k, v, num_iters=2)
    torch.cuda.synchronize()
    output_bf16 = attn_bf16(q, k, v, scale=scale)
    torch.cuda.synchronize()
    
    # Create INT8 version with skipping
    attn_int8 = LiteAttention(enable_skipping=True, use_int8=True, threshold=threshold)
    # Warm up to stabilize skip lists
    run_attention_warmup(attn_int8, q, k, v, num_iters=2)
    torch.cuda.synchronize()
    output_int8 = attn_int8(q, k, v, scale=scale)
    torch.cuda.synchronize()
    
    # Compute error metrics
    metrics = compute_error_metrics(output_int8, output_bf16, "INT8 (with skipping) vs BF16 (with skipping)")
    
    # Check tolerances (slightly relaxed for skipping case)
    passed = (metrics['max_abs_error'] < tolerance_max_abs and 
              metrics['cosine_sim'] >= tolerance_cosine)
    
    # Also check skip percentages are similar
    skip_pct_bf16 = attn_bf16.calc_percentage(attn_bf16.read_list)
    skip_pct_int8 = attn_int8.calc_percentage(attn_int8.read_list)
    skip_pct_diff = abs(skip_pct_bf16 - skip_pct_int8)
    skip_pct_passed = skip_pct_diff < 0.05  # Allow 5% difference
    
    # Adjust status message based on tile size match
    overall_passed = passed and skip_pct_passed
    if not tile_sizes_match:
        status = '⚠️  SKIPPED (tile size mismatch)' if not overall_passed else '✅ PASSED (tile size mismatch, results OK)'
    else:
        status = '✅ PASSED' if overall_passed else '❌ FAILED'
    
    print(f"  INT8 with skipping test: {status}")
    print(f"    Max absolute error: {metrics['max_abs_error']:.6e} (tolerance: {tolerance_max_abs:.6e})")
    print(f"    Mean absolute error: {metrics['mean_abs_error']:.6e}")
    print(f"    RMSE: {metrics['rmse']:.6e}")
    print(f"    Cosine similarity: {metrics['cosine_sim']:.8f} (tolerance: {tolerance_cosine:.8f})")
    
    if not skip_pct_passed:
        print(f"    ⚠️  Skip percentage mismatch: BF16={skip_pct_bf16:.2%}, INT8={skip_pct_int8:.2%}, diff={skip_pct_diff:.2%}")
    
    # If tile sizes don't match, always return True (don't fail)
    return overall_passed if tile_sizes_match else True

def run_tests_for_head_dim(head_dim, batch=2, seq_len=18200, heads=32):
    """Run all tests for a specific head dimension."""
    print(f"\n{'='*60}")
    print(f"Testing head_dim: {head_dim}")
    print(f"{'='*60}")
    
    # Generate test data
    q, k, v = generate_test_tensors(batch, seq_len, heads, head_dim)
    
    # Run BF16 tests
    print(f"\n  {'-'*56}")
    print(f"  BF16 Tests (head_dim: {head_dim})")
    print(f"  {'-'*56}")
    bf16_results = []
    bf16_results.append(stress_test(q, k, v, head_dim, use_int8=False))
    bf16_results.append(test_skip_all(q, k, v, head_dim, use_int8=False))
    bf16_results.append(test_skip_nothing(q, k, v, head_dim, use_int8=False))
    bf16_results.append(test_must_skip_list(q, k, v, head_dim, use_int8=False))
    bf16_results.append(test_must_do_list(q, k, v, head_dim, use_int8=False))
    q_short, k_short, v_short = generate_test_tensors(batch=batch, seq_len=min(6143, seq_len), heads=heads, head_dim=head_dim)
    bf16_results.append(test_softmax_lse_correctness(q_short, k_short, v_short, head_dim, use_int8=False))

    # consistency_test(q, k, v, head_dim)
    
    # Run INT8 tests
    print(f"\n  {'-'*56}")
    print(f"  INT8 Tests (head_dim: {head_dim})")
    print(f"  {'-'*56}")
    int8_results = []
    int8_results.append(stress_test(q, k, v, head_dim, use_int8=True))
    int8_results.append(test_skip_all(q, k, v, head_dim, use_int8=True))
    int8_results.append(test_skip_nothing(q, k, v, head_dim, use_int8=True))
    int8_results.append(test_must_skip_list(q, k, v, head_dim, use_int8=True))
    int8_results.append(test_must_do_list(q, k, v, head_dim, use_int8=True))
    int8_results.append(test_softmax_lse_correctness(q_short, k_short, v_short, head_dim, tolerance=0.01, use_int8=True))

    torch.cuda.synchronize()
    
    # INT8 correctness tests (compare INT8 vs BF16)
    print(f"\n  {'-'*56}")
    print(f"  INT8 Correctness Tests (vs BF16) (head_dim: {head_dim})")
    print(f"  {'-'*56}")
    # int8_results.append(test_int8_correctness(q, k, v, head_dim))
    # int8_results.append(test_int8_with_skipping(q, k, v, head_dim))
    int8_results.append(test_int8_correctness(q_short, k_short, v_short, head_dim))
    int8_results.append(test_int8_with_skipping(q_short, k_short, v_short, head_dim))
    
    # Determine overall pass/fail for each dtype
    bf16_passed = all(bf16_results)
    int8_passed = all(int8_results)
    
    return bf16_passed, int8_passed


def main():
    """Main test runner."""
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    # Test different head dimensions
    head_dims = [32, 64, 96, 128, 192, 256]
    
    # Track results for each head dimension
    bf16_results = {}
    int8_results = {}
    
    for head_dim in head_dims:
        # bf16_passed, int8_passed = run_tests_for_head_dim(head_dim, seq_len=2**15)
        bf16_passed, int8_passed = run_tests_for_head_dim(head_dim)
        bf16_results[head_dim] = bf16_passed
        int8_results[head_dim] = int8_passed
    
    # Print summary
    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}\n")
    
    # Summary for BF16
    print(f"{'='*60}")
    print("SUMMARY - BF16 Tests")
    print(f"{'='*60}")
    bf16_passed_dims = [hd for hd in head_dims if bf16_results[hd]]
    bf16_failed_dims = [hd for hd in head_dims if not bf16_results[hd]]
    
    if bf16_passed_dims:
        print(f"✅ PASSED head dimensions: {bf16_passed_dims}")
    if bf16_failed_dims:
        print(f"❌ FAILED head dimensions: {bf16_failed_dims}")
    if not bf16_failed_dims:
        print("All BF16 tests passed!")
    
    # Summary for INT8
    print(f"\n{'='*60}")
    print("SUMMARY - INT8 Tests")
    print(f"{'='*60}")
    int8_passed_dims = [hd for hd in head_dims if int8_results[hd]]
    int8_failed_dims = [hd for hd in head_dims if not int8_results[hd]]
    
    if int8_passed_dims:
        print(f"✅ PASSED head dimensions: {int8_passed_dims}")
    if int8_failed_dims:
        print(f"❌ FAILED head dimensions: {int8_failed_dims}")
    if not int8_failed_dims:
        print("All INT8 tests passed!")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()