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


def assert_valid_MNK(m, n, k, q_dtype, k_dtype, v_dtype):
    assert m % 64 == 0, "m must be divisible by 64"
    assert n % 8 == 0, "n must be divisible by 16"
    if q_dtype == torch.int8:
        assert k % 32 == 0, "k must be divisible by 32"
    else:
        assert k % 16 == 0, "k must be divisible by 16"


# Returns the number of BYTES needed for registers and shared memory
def mem_calc(m, n, k, q_dtype=torch.int8, k_dtype=torch.int8, v_dtype=torch.float16, 
             MmaPV_is_RS=False, isIntraWGOverlap=False):
    assert_valid_MNK(m, n, k, q_dtype, k_dtype, v_dtype)
    q_size = memQ(m, n, k, q_dtype)
    k_size = memK(m, n, k, k_dtype)
    v_size = memV(m, n, k, v_dtype)
    qk_size = memQK(m, n, k, torch.int32)
    p_size = memP(m, n, k, torch.float16)
    o_size = memO(m, n, k, torch.float32)

    rmem = qk_size + o_size  # the results of wgmma op always in registers
    smem = q_size + k_size + v_size  # we always hold Q, K, V in smem

    if MmaPV_is_RS:
        rmem += p_size * isIntraWGOverlap
    else:
        smem += p_size * isIntraWGOverlap

    return torch.tensor([rmem, smem])


def mem_analysis(m, n, k, q_dtype=torch.int8, k_dtype=torch.int8, v_dtype=torch.float16, 
                 MmaPV_is_RS=False, isIntraWGOverlap=False):
    mem_bytes = mem_calc(m, n, k, q_dtype, k_dtype, v_dtype, MmaPV_is_RS, isIntraWGOverlap)
    rmem, smem = mem_bytes
    ratio = mem_bytes / torch.tensor([rmem_limit, smem_limit])
    print(f"rmem/rmem_limit: {ratio[0]:.2f}, smem/smem_limit: {ratio[1]:.2f}")
    num_wg = m // 64
    registers_per_mma_thread = rmem / (WARP_SIZE * BYTES_PER_REGISTER * WARPS_PER_WG * num_wg)
    print(f"registers needed to be allocated for each mma thread: {int(registers_per_mma_thread)}")

    remaining_registers = (rmem_per_sub_partition - (rmem / NUM_SUB_PARTITIONS)) / BYTES_PER_REGISTER
    print(f"remaining registers per sub-partition: {int(remaining_registers)}")
    remaining_smem = smem_limit - smem
    print(f"remaining smem in bytes: {int(remaining_smem)}")

    # tensor cores operations for QK and PV (per warpgroup)
    num_ts_ops_qk = k // (32 if q_dtype.itemsize == 1 else 16)
    num_ts_ops_pv = k // (32 if v_dtype.itemsize == 1 else 16)
    


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

