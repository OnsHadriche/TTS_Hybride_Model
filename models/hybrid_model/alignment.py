import numpy as np
from numba import jit, prange

@jit(nopython=True)
def mas(log_attn_map, width=1):
    """
    Performs monotonic alignment search (MAS) on a log attention map.
    Assumes an input of shape (mel, text).

    Args:
        log_attn_map (np.ndarray): Logarithmic attention map with shape (mel, text).
        width (int): Local window width. Default is 1.

    Returns:
        np.ndarray: A binary (0/1) alignment matrix of the same shape as log_attn_map.
    """
    opt = np.zeros_like(log_attn_map)
    log_attn_map = log_attn_map.copy()
    # Force log attention for text indices >0 (at the first time step) to be -inf.
    log_attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(log_attn_map)
    log_p[0, :] = log_attn_map[0, :]
    prev_ind = np.zeros_like(log_attn_map, dtype=np.int64)
    # Dynamic programming forward pass.
    for i in range(1, log_attn_map.shape[0]):
        for j in range(log_attn_map.shape[1]):
            prev_j = np.arange(max(0, j - width), j + 1)
            prev_log = np.array([log_p[i - 1, prev_idx] for prev_idx in prev_j])
            ind = np.argmax(prev_log)
            log_p[i, j] = log_attn_map[i, j] + prev_log[ind]
            prev_ind[i, j] = prev_j[ind]
    # Backtracking to obtain the optimal alignment path.
    curr_text_idx = log_attn_map.shape[1] - 1
    for i in range(log_attn_map.shape[0] - 1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1
    return opt


@jit(nopython=True)
def mas_width1(log_attn_map):
    """
    Monotonic alignment search (MAS) with a hardcoded width=1.
    Assumes an input of shape (mel, text).

    Args:
        log_attn_map (np.ndarray): Logarithmic attention map with shape (mel, text).

    Returns:
        np.ndarray: A binary (0/1) alignment matrix.
    """
    neg_inf = log_attn_map.dtype.type(-np.inf)
    log_p = log_attn_map.copy()
    log_p[0, 1:] = neg_inf
    for i in range(1, log_p.shape[0]):
        prev_log1 = neg_inf
        for j in range(log_p.shape[1]):
            prev_log2 = log_p[i - 1, j]
            log_p[i, j] += max(prev_log1, prev_log2)
            prev_log1 = prev_log2

    # Backtracking to form the alignment output.
    opt = np.zeros_like(log_p)
    one = opt.dtype.type(1)
    j = log_p.shape[1] - 1
    for i in range(log_p.shape[0] - 1, 0, -1):
        opt[i, j] = one
        if log_p[i - 1, j - 1] >= log_p[i - 1, j]:
            j -= 1
            if j == 0:
                opt[1:i, j] = one
                break
    opt[0, j] = one
    return opt


@jit(nopython=True, parallel=True)
def b_mas(b_log_attn_map, in_lens, out_lens, width=1):
    """
    Batched monotonic alignment search (MAS) over a batch of log attention maps.
    Each attention map is of shape (1, out_len, in_len) per batch element.

    Args:
        b_log_attn_map (np.ndarray): Batch of log attention maps with shape (B, 1, out_max, in_max).
        in_lens (np.ndarray): Array of input lengths (for text) for each batch element.
        out_lens (np.ndarray): Array of output lengths (for mel) for each batch element.
        width (int): Local window width. Default is 1. Must be 1.

    Returns:
        np.ndarray: Batch of binary alignment matrices with shape (B, 1, out_max, in_max).
    """
    assert width == 1
    attn_out = np.zeros_like(b_log_attn_map)
    for b in prange(b_log_attn_map.shape[0]):
        # Process each batch element using mas_width1 over the actual lengths.
        out = mas_width1(b_log_attn_map[b, 0, :out_lens[b], :in_lens[b]])
        attn_out[b, 0, :out_lens[b], :in_lens[b]] = out
    return attn_out
