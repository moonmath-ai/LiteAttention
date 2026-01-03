import torch

# Constants
KSTAGES = 2
WARP_SIZE = 32
BYTES_PER_REGISTER = 4
WARPS_PER_WG = 4
NUM_SUB_PARTITIONS = 4

smem_limit = 256000  # 256KB
rmem_per_sub_partition = (2**14) * BYTES_PER_REGISTER
rmem_limit = rmem_per_sub_partition * NUM_SUB_PARTITIONS

limits = torch.tensor([rmem_limit, smem_limit])
ratio_limit = torch.tensor([0.9, 1])  # it's 0.9 because we need tiny amount of extra registers for non mma operations

# Configuration search parameters
possible_m = [64, 128, 192]
possible_n = 16 * torch.arange(4, 16)
possible_MmaPV_is_RS = [False, True]
possible_isIntraWGOverlap = [False, True]


# Memory size calculation functions
def memQ(m, n, k, dtype):
    return m * k * dtype.itemsize


def memK(m, n, k, dtype):
    return n * k * KSTAGES * dtype.itemsize


def memV(m, n, k, dtype):
    return n * k * KSTAGES * dtype.itemsize


def memQK(m, n, k, dtype):
    return m * n * dtype.itemsize


def memP(m, n, k, dtype):
    return m * n * dtype.itemsize


def memO(m, n, k, dtype):
    return m * k * dtype.itemsize


def granularity_for_dtype(dtype):
    if dtype.itemsize == 1:
        return 32
    else:
        return 16

def assert_valid_dim(dim, divisible_by, dim_name):
    assert dim % divisible_by == 0, dim_name + " must be divisible by " + str(divisible_by)

def assert_valid_m(m, dim_name, strictly_two_wg=False):
    assert_valid_dim(m, 128 if strictly_two_wg else 64, dim_name)

def assert_valid_n(n, dim_name):
    assert_valid_dim(n, 8, dim_name)

def assert_valid_k(k, dtype, dim_name):
    assert_valid_dim(k, granularity_for_dtype(dtype), dim_name)

def assert_valid_MNK(m, n, k, q_dtype, k_dtype, v_dtype, strictly_two_wg=False):
    assert_valid_m(m, "m")
    assert_valid_n(n, "n")
    assert_valid_k(k, q_dtype, "k")

    assert_valid_k(n, v_dtype, "n also (because of PV)")


# Returns the number of BYTES needed for registers and shared memory
def mem_calc(m, n, k, q_dtype=torch.int8, k_dtype=torch.int8, v_dtype=torch.float16, 
             MmaPV_is_RS=False, isIntraWGOverlap=False, strictly_two_wg=False):

    q_size = memQ(m, n, k, q_dtype)
    k_size = memK(m, n, k, k_dtype)
    v_size = memV(m, n, k, v_dtype)
    # qk_size = memQK(m, n, k, torch.int32)
    
    # m_for_qk = 64 * 2 if strictly_two_wg else m
    m_for_qk = m // 2 if strictly_two_wg else m
    qk_size = memQK(m_for_qk, n, k, torch.int32)
    p_size = memP(m, n, k, torch.bfloat16)
    o_size = memO(m, n, k, torch.float32)

    rmem = qk_size + o_size  # the results of wgmma op always in registers
    smem = q_size + k_size + v_size  # we always hold Q, K, V in smem

    if MmaPV_is_RS:
        rmem += p_size * isIntraWGOverlap
    else:
        smem += p_size * isIntraWGOverlap

    return torch.tensor([rmem, smem])


def mem_analysis(m, n, k, q_dtype=torch.int8, k_dtype=torch.int8, v_dtype=torch.float16, 
                 MmaPV_is_RS=False, isIntraWGOverlap=False, strictly_two_wg=False):
    mem_bytes = mem_calc(m, n, k, q_dtype, k_dtype, v_dtype, MmaPV_is_RS, isIntraWGOverlap, strictly_two_wg)
    rmem, smem = mem_bytes
    ratio = mem_bytes / torch.tensor([rmem_limit, smem_limit])
    print(f"rmem/rmem_limit: {ratio[0]:.2f}, smem/smem_limit: {ratio[1]:.2f}")
    # num_wg = m // 64
    num_wg = 2 if strictly_two_wg else m // 64
    registers_per_mma_thread = rmem / (WARP_SIZE * BYTES_PER_REGISTER * WARPS_PER_WG * num_wg)
    print(f"registers needed to be allocated for each mma thread: {int(registers_per_mma_thread)}")

    remaining_registers = (rmem_per_sub_partition - (rmem / NUM_SUB_PARTITIONS)) / BYTES_PER_REGISTER
    print(f"remaining registers per sub-partition: {int(remaining_registers)}")
    remaining_smem = smem_limit - smem
    print(f"remaining smem in bytes: {int(remaining_smem)}")

    # tensor cores operations for QK and PV (per warpgroup)
    granularity_qk = granularity_for_dtype(q_dtype)
    num_ts_ops_qk = (k + granularity_qk - 1) // granularity_qk
    num_ts_ops_strictly_two_wg_string = f"{m // 128} X " if strictly_two_wg else ""
    print(f"tensor cores operations for QK: {num_ts_ops_strictly_two_wg_string}{num_ts_ops_qk} X m{64}n{n}k{granularity_qk}")

    granularity_pv = granularity_for_dtype(v_dtype)
    num_ts_ops_pv = (n + granularity_pv - 1) // granularity_pv
    print(f"tensor cores operations for PV: {num_ts_ops_strictly_two_wg_string}{num_ts_ops_pv} X m{64}n{k}k{granularity_pv}")

    assert_valid_MNK(m, n, k, q_dtype, k_dtype, v_dtype, strictly_two_wg)
    


def mem_analysis_table(m, n, k, q_dtype=torch.int8, k_dtype=torch.int8, v_dtype=torch.float16):
    """
    Print an analysis table for all combinations of MmaPV_is_RS, isIntraWGOverlap, and strictly_two_wg.
    """
    # Collect data for all combinations
    rows = []
    
    for MmaPV_is_RS in [False, True]:
        for isIntraWGOverlap in [False, True]:
            for strictly_two_wg in [False, True]:
                try:
                    mem_bytes = mem_calc(m, n, k, q_dtype, k_dtype, v_dtype, MmaPV_is_RS, isIntraWGOverlap, strictly_two_wg)
                    rmem, smem = mem_bytes
                    ratio = mem_bytes / torch.tensor([rmem_limit, smem_limit])
                    
                    num_wg = 2 if strictly_two_wg else m // 64
                    registers_per_mma_thread = rmem / (WARP_SIZE * BYTES_PER_REGISTER * WARPS_PER_WG * num_wg)
                    
                    remaining_registers = (rmem_per_sub_partition - (rmem / NUM_SUB_PARTITIONS)) / BYTES_PER_REGISTER
                    remaining_smem = smem_limit - smem
                    
                    granularity_qk = granularity_for_dtype(q_dtype)
                    num_ts_ops_qk = (k + granularity_qk - 1) // granularity_qk
                    num_ts_ops_strictly_two_wg_string = f"{m // 128} X " if strictly_two_wg else ""
                    qk_ops_str = f"{num_ts_ops_strictly_two_wg_string}{num_ts_ops_qk} X m{64}n{n}k{granularity_qk}"
                    
                    granularity_pv = granularity_for_dtype(v_dtype)
                    num_ts_ops_pv = (n + granularity_pv - 1) // granularity_pv
                    pv_ops_str = f"{num_ts_ops_strictly_two_wg_string}{num_ts_ops_pv} X m{64}n{k}k{granularity_pv}"
                    
                    rows.append({
                        'MmaPV_is_RS': MmaPV_is_RS,
                        'isIntraWGOverlap': isIntraWGOverlap,
                        'strictly_two_wg': strictly_two_wg,
                        'rmem_ratio': ratio[0].item(),
                        'smem_ratio': ratio[1].item(),
                        'registers_per_mma_thread': int(registers_per_mma_thread),
                        'remaining_registers': int(remaining_registers),
                        'remaining_smem': int(remaining_smem),
                        'qk_ops': qk_ops_str,
                        'pv_ops': pv_ops_str,
                        'valid': True
                    })
                except AssertionError:
                    rows.append({
                        'MmaPV_is_RS': MmaPV_is_RS,
                        'isIntraWGOverlap': isIntraWGOverlap,
                        'strictly_two_wg': strictly_two_wg,
                        'rmem_ratio': None,
                        'smem_ratio': None,
                        'registers_per_mma_thread': None,
                        'remaining_registers': None,
                        'remaining_smem': None,
                        'qk_ops': None,
                        'pv_ops': None,
                        'valid': False
                    })
    
    # Print table header
    print(f"\nMemory Analysis Table for m={m}, n={n}, k={k}")
    print(f"q_dtype={q_dtype}, k_dtype={k_dtype}, v_dtype={v_dtype}\n")
    
    # Format table
    header = f"{'MmaPV_is_RS':<12} {'isIntraWG':<10} {'strict_2wg':<10} {'rmem/lim':<10} {'smem/lim':<10} {'reg/mma':<10} {'rem_reg':<10} {'rem_smem':<12} {'QK ops':<25} {'PV ops':<25}"
    print(header)
    print("-" * len(header))
    
    for row in rows:
        if row['valid']:
            mma_str = str(row['MmaPV_is_RS'])
            intra_str = str(row['isIntraWGOverlap'])
            strict_str = str(row['strictly_two_wg'])
            rmem_str = f"{row['rmem_ratio']:.2f}"
            smem_str = f"{row['smem_ratio']:.2f}"
            reg_str = str(row['registers_per_mma_thread'])
            rem_reg_str = str(row['remaining_registers'])
            rem_smem_str = str(row['remaining_smem'])
            qk_str = row['qk_ops']
            pv_str = row['pv_ops']
        else:
            mma_str = str(row['MmaPV_is_RS'])
            intra_str = str(row['isIntraWGOverlap'])
            strict_str = str(row['strictly_two_wg'])
            rmem_str = "INVALID"
            smem_str = "INVALID"
            reg_str = "INVALID"
            rem_reg_str = "INVALID"
            rem_smem_str = "INVALID"
            qk_str = "INVALID"
            pv_str = "INVALID"
        
        print(f"{mma_str:<12} {intra_str:<10} {strict_str:<10} {rmem_str:<10} {smem_str:<10} {reg_str:<10} {rem_reg_str:<10} {rem_smem_str:<12} {qk_str:<25} {pv_str:<25}")
    
    print()


def config_score(m, n, k, MmaPV_is_RS=False, isIntraWGOverlap=False):
    mem_bytes = mem_calc(m=m, n=n, k=k, MmaPV_is_RS=MmaPV_is_RS, isIntraWGOverlap=isIntraWGOverlap)
    rmem, smem = mem_bytes
    ratio = mem_bytes / limits
    if (ratio > ratio_limit).any():
        return -torch.inf
    
    # rescale because rmem is more valuable than smem
    ratio_rescaled = ratio * torch.tensor([25, 1])
    return ratio_rescaled.norm(p=2) + (torch.tensor(1e-6) * isIntraWGOverlap) + (torch.tensor(1e-6) * MmaPV_is_RS)


def top_k_configs(k, top_k=3):
    configs = []
    
    # Iterate over all possible combinations
    for m in possible_m:
        for n in possible_n:
            n_val = n.item() if isinstance(n, torch.Tensor) else n
            for MmaPV_is_RS in possible_MmaPV_is_RS:
                for isIntraWGOverlap in possible_isIntraWGOverlap:
                    score = config_score(m, n_val, k, MmaPV_is_RS, isIntraWGOverlap)
                    # Only include valid configs (score != -inf)
                    score_val = score.item() if isinstance(score, torch.Tensor) else score
                    configs.append({
                        'm': m,
                        'n': n_val,
                        'k': k,
                        'MmaPV_is_RS': MmaPV_is_RS,
                        'isIntraWGOverlap': isIntraWGOverlap,
                        'score': score_val
                    })
    
    # Sort by score (lower is better)
    configs.sort(key=lambda x: x['score'], reverse=True)
    
    # Return top k configs
    return configs[:top_k]

