import torch

def ceil_div(x, y):
    return (x + y - 1) // y

def init_skip_list(q):
    batch, seq_len, heads, head_dim = q.shape
    ceil_div = lambda x, y: (x + y - 1) // y
    qtiles = ceil_div(seq_len, 128)
    ktiles = ceil_div(seq_len, 176)
    skip_list = torch.zeros(2, batch, heads, qtiles, ktiles + 1, dtype=torch.uint32)
    skip_list[0, :, :, 2] = ktiles
    skip_list[0, :, :, 0] = 2

    return skip_list

def skip_list_update(skip_list_read, skip_list_write, rule_mask):

    list_len = skip_list_read[0]
    write_idx = 1
    is_skipping = True

    for read_idx in range(1, list_len + 1, 2):
        start_idx = skip_list_read[read_idx]
        end_idx = skip_list_read[read_idx + 1]
        for i in range(start_idx, end_idx):
            skip = rule_mask[i]
            if skip != is_skipping:
                skip_list_write[write_idx] = i
                is_skipping = skip
                write_idx += 1

        is_skipping = True
        if skip != is_skipping:
            skip_list_write[write_idx] = end_idx
            write_idx += 1

    skip_list_write[0] = write_idx - 1