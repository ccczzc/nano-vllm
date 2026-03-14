import triton 
import triton.language as tl
from nanovllm.utils.context import get_context
import torch
import torch.nn as nn

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

@triton.jit
def flash_attention_prefill_ordinary_kernel(
    Q, K, V, O,
    cu_seqlens_q_ptr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    """
    Flash Attention kernel for variable-length sequences.
    Each program processes one block of queries for one head in one sequence.
    """
    # Program IDs
    q_tile_index = tl.program_id(0) # block index
    off_h = tl.program_id(1) # head index
    seq_idx = tl.program_id(2) # sequence index

    # Determine which KV head to use (for GQA)
    kv_head_idx = off_h // (num_heads // num_kv_heads)
    
    # Load sequence boundaries
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seq_len = seq_end - seq_start
    
    # Early exit if this block is beyond sequence length
    if q_tile_index * Q_TILE_SIZE >= seq_len:
        return
    
    # Offset for this block of queries
    offs_m = q_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    offs_d = tl.arange(0, head_dim)
    
    # Query pointers: Q has shape (total_tokens, num_heads, head_dim)
    q_ptrs = Q + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    
    # Load Q block - shape (BLOCK_M, head_dim)
    mask_m = offs_m < seq_len
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    q = (q * scale).to(q.dtype)
    
    # Initialize output accumulators
    l_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    m_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32) - 1e10
    acc = tl.zeros([Q_TILE_SIZE, head_dim], dtype=tl.float32)
    
    # Number of blocks to process
    num_blocks = tl.cdiv(seq_len, K_TILE_SIZE)
    
    # Loop over K, V blocks
    for block_n in range(num_blocks):
        start_n = block_n * K_TILE_SIZE
        offs_n = start_n + tl.arange(0, K_TILE_SIZE)
        
        # Mask for valid positions
        mask_n = offs_n < seq_len
        
        # K pointers: K has shape (total_tokens, num_kv_heads, head_dim)
        k_ptrs = K + (seq_start + offs_n[None, :]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[:, None]
        
        # Load K block - shape (head_dim, BLOCK_N)
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        
        # Compute QK^T - shape (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, k)
        
        # Apply causal mask: only attend to positions <= current position
        mask_causal = (offs_m[:, None] + seq_start) >= (offs_n[None, :] + seq_start)
        qk = tl.where(mask_causal & mask_n[None, :], qk, -1e10)
        
        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        # Rescale previous accumulator
        acc = acc * alpha[:, None]
        
        # Load V block - shape (BLOCK_N, head_dim)
        v_ptrs = V + (seq_start + offs_n[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Accumulate weighted values
        acc = acc + tl.dot(p.to(v.dtype), v)
        
        # Update normalizer
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output: O has shape (total_tokens, num_heads, head_dim)
    o_ptrs = O + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


def flash_attention_prefill_ordinary(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Optimized Flash Attention for prefill phase with variable-length sequences.
    
    Args:
        q: (total_tokens, num_heads, head_dim)
        k: (total_tokens, num_kv_heads, head_dim)
        v: (total_tokens, num_kv_heads, head_dim)
        cu_seqlens_q: cumulative sequence lengths for queries
        max_seqlen_q: maximum sequence length for queries
        scale: attention scale factor
    
    Returns:
        output: (total_tokens, num_heads, head_dim)
    """
    # Make tensors contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # Allocate output
    output = torch.empty_like(q)
    
    if head_dim <= 64:
        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64
    elif head_dim <= 128:
        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32
    else:
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16
    
    # Number of sequences
    num_seqs = cu_seqlens_q.shape[0] - 1
    
    # Calculate grid dimensions - launch all kernels at once
    grid = (triton.cdiv(max_seqlen_q, Q_TILE_SIZE), num_heads, num_seqs)
    
    flash_attention_prefill_ordinary_kernel[grid](
        q, k, v, output,
        cu_seqlens_q,
        scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        Q_TILE_SIZE=Q_TILE_SIZE,
        K_TILE_SIZE=K_TILE_SIZE,
    )
    
    return output

@triton.jit
def flash_attention_paged_prefill_kernel(
    Q, 
    K_CACHE, 
    V_CACHE, 
    O,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    block_tables_ptr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    """
    Paged Attention Prefill kernel!
    Supports calculation with KV Cache prefix lookup table.
    """
    assert K_TILE_SIZE == block_size, "K_TILE_SIZE must match block_size limit for exact mapping"
    
    q_tile_index = tl.program_id(0) # block index
    off_h = tl.program_id(1)        # head index
    seq_idx = tl.program_id(2)      # sequence index

    kv_head_idx = off_h // (num_heads // num_kv_heads)
    
    # Get boundaries for Q (total_tokens_q, num_heads, head_dim)
    seq_start_q = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end_q = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seq_len_q = seq_end_q - seq_start_q

    # Get boundaries for the complete KV (total_tokens_k, num_kv_heads, head_dim)
    seq_start_k = tl.load(cu_seqlens_k_ptr + seq_idx) 
    seq_end_k = tl.load(cu_seqlens_k_ptr + seq_idx + 1)
    seq_len_k = seq_end_k - seq_start_k

    # Number of tokens that don't need calculation due to prefix cache hits
    history_len = seq_len_k - seq_len_q 
    
    # Grid boundary check (based on the total length of the new Q)
    if q_tile_index * Q_TILE_SIZE >= seq_len_q:
        return
    
    # Calculate relative offsets for Q matrix (Shape: [Q_TILE_SIZE, head_dim])
    offs_q = q_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    # Global absolute position in the sequence for causal masking comparison
    absolute_pos_q = offs_q + history_len   
    offs_d = tl.arange(0, head_dim)
    
    q_ptrs = Q + (seq_start_q + offs_q[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    
    mask_m = offs_q < seq_len_q
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    q = (q * scale).to(q.dtype)
    
    # Initialize accumulators
    l_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    m_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32) - 1e10
    acc = tl.zeros([Q_TILE_SIZE, head_dim], dtype=tl.float32)
    
    # Main loop over the full K length (including history)
    num_blocks = tl.cdiv(seq_len_k, K_TILE_SIZE)
    
    # Paged KV Cache main loop
    for block_n in range(num_blocks):
        start_n = block_n * K_TILE_SIZE
        offs_n = start_n + tl.arange(0, K_TILE_SIZE)
        
        # Boundary check compares against complete history length
        mask_n = offs_n < seq_len_k
        
        # Look up physical block from block table
        table_offset = seq_idx * max_num_blocks + block_n
        physical_block_idx = tl.load(block_tables_ptr + table_offset)
        
        
        physical_block_idx_safe = tl.where(physical_block_idx < 0, 0, physical_block_idx)

        # K Block shape is (head_dim, K_TILE_SIZE), 
        # with TILE on dimension 1 and head_dim on dimension 0
        k_ptrs = K_CACHE + (
            physical_block_idx_safe * block_size * num_kv_heads * head_dim + 
            tl.arange(0, K_TILE_SIZE)[None, :] * num_kv_heads * head_dim +  # TILE on dim 1
            kv_head_idx * head_dim + 
            offs_d[:, None]                                                 # dim on dim 0
        )
        
        valid_load_mask = mask_n[None, :] & (physical_block_idx >= 0)
        k = tl.load(k_ptrs, mask=valid_load_mask, other=0.0)
        
        # Score calculation: [Q_TILE_SIZE, head_dim] x [head_dim, K_TILE_SIZE] = [Q_TILE_SIZE, K_TILE_SIZE]
        qk = tl.dot(q, k)
        
        # Causal mask: absolute Q position >= absolute K position
        mask_causal = absolute_pos_q[:, None] >= offs_n[None, :]
        qk = tl.where(mask_causal & mask_n[None, :], qk, -1e10)
        
        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        acc = acc * alpha[:, None]
        
        # V Block shape is (K_TILE_SIZE, head_dim), orientation same as Q
        v_ptrs = V_CACHE + (
            physical_block_idx_safe * block_size * num_kv_heads * head_dim + 
            tl.arange(0, K_TILE_SIZE)[:, None] * num_kv_heads * head_dim +  # TILE on dim 0
            kv_head_idx * head_dim + 
            offs_d[None, :]                                                 # dim on dim 1
        )
        
        valid_load_mask_v = mask_n[:, None] & (physical_block_idx >= 0)
        v = tl.load(v_ptrs, mask=valid_load_mask_v, other=0.0)
        
        # Accumulate results
        acc = acc + tl.dot(p.to(v.dtype), v)
        
        # Update normalizer
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new
    
    # Normalize and write output
    acc = acc / l_i[:, None]
    
    o_ptrs = O + (seq_start_q + offs_q[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


def flash_attention_paged_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_tables: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int
) -> torch.Tensor:
    
    q = q.contiguous()
    output = torch.empty_like(q)
    
    # Strictly align K_TILE = block_size
    K_TILE_SIZE = block_size
    Q_TILE_SIZE = 64 if head_dim <= 64 else 32
    
    num_seqs = cu_seqlens_q.shape[0] - 1
    cu_seqlens_cpu = cu_seqlens_q.cpu()
    max_seq_len_q = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item()
    
    # Grid must be launched based on Q length, as that's the number of tokens to compute
    grid = (triton.cdiv(max_seq_len_q, Q_TILE_SIZE), num_heads, num_seqs)
    
    flash_attention_paged_prefill_kernel[grid](
        q, k_cache, v_cache, output,
        cu_seqlens_q, cu_seqlens_k, block_tables,
        scale, num_heads, num_kv_heads, head_dim,
        block_size, max_num_blocks=block_tables.shape[1],
        Q_TILE_SIZE=Q_TILE_SIZE,
        K_TILE_SIZE=K_TILE_SIZE,
    )
    
    return output


@triton.jit
def paged_attention_decode_block_kernel(
    output_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
):
    """
    Optimized block-level Paged Attention decode kernel.
    """
    seq_idx = tl.program_id(0)    # batch index
    head_idx = tl.program_id(1)   # head index
    
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    context_len = tl.load(context_lens_ptr + seq_idx)
    
    # Get Q pointers and load: shape (head_dim,)
    offs_d = tl.arange(0, head_dim)
    q_offset = seq_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset)
    q = (q * scale).to(q.dtype)
    
    # Initialize online softmax and accumulator
    m_i = -1e10
    l_i = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)
    
    # Calculate number of blocks to traverse
    num_blocks = tl.cdiv(context_len, block_size)
    offs_block = tl.arange(0, block_size)
    
    for block_idx in range(num_blocks):
        # Look up physical block from block table
        physical_block_idx = tl.load(block_tables_ptr + seq_idx * max_num_blocks + block_idx)
        
        # Protection mechanism against invalid blocks
        physical_block_idx_safe = tl.where(physical_block_idx < 0, 0, physical_block_idx)

        # Build pointer for K block: load entire physical block (block_size, head_dim)
        k_offset = (
            physical_block_idx_safe * block_size * num_kv_heads * head_dim +
            offs_block[:, None] * num_kv_heads * head_dim +  # (block_size, 1)
            kv_head_idx * head_dim +
            offs_d[None, :]                                  # (1, head_dim)
        )
        
        # Mask for valid tokens within the physical block (last block may be partial)
        token_indices = block_idx * block_size + offs_block
        mask_n = (token_indices < context_len) & (physical_block_idx >= 0)
        
        # Load K block and compute scores - (block_size, head_dim)
        k_block = tl.load(k_cache_ptr + k_offset, mask=mask_n[:, None], other=0.0)
        
        # Dot product of Q and current K block (Broadcasting Q, aggregating along dimension 1)
        # [block_size, head_dim] * [1, head_dim] -> [block_size, head_dim] -> sum -> [block_size]
        qk = tl.sum(q[None, :] * k_block, axis=1)
        qk = tl.where(mask_n, qk, -1e10)
        
        # Online Softmax update
        m_ij = tl.max(qk, axis=0)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new)
        
        # Update accumulator smoothly
        acc = acc * alpha
        l_i = l_i * alpha + tl.sum(tl.where(mask_n, p, 0.0), axis=0)
        
        # Load V block - (block_size, head_dim)
        v_offset = (
            physical_block_idx_safe * block_size * num_kv_heads * head_dim +
            offs_block[:, None] * num_kv_heads * head_dim +
            kv_head_idx * head_dim +
            offs_d[None, :]
        )
        v_block = tl.load(v_cache_ptr + v_offset, mask=mask_n[:, None], other=0.0)
        
        # [block_size] * [block_size, head_dim] -> [block_size, head_dim]
        # Aggregate across columns for valid p values
        p_expanded = tl.where(mask_n[:, None], p[:, None], 0.0)
        acc += tl.sum(p_expanded * v_block, axis=0)
        
        m_i = m_i_new
    
    # Normalize and write back output
    out = acc / l_i
    out_offset = seq_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    tl.store(output_ptr + out_offset, out.to(query_ptr.dtype.element_ty))


def paged_attention_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int
) -> torch.Tensor:
    """
    Compute attention in decode mode using paged KV cache.
    
    Args:
        query: (batch_size, num_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        block_tables: (batch_size, max_num_blocks)
        context_lens: (batch_size,)
        scale: attention scale factor
    
    Returns:
        output: (batch_size, num_heads, head_dim)
    """
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]
    
    # Make contiguous
    query = query.contiguous()
    
    output = torch.empty_like(query)
    
    grid = (batch_size, num_heads)
    
    paged_attention_decode_block_kernel[grid](
        output,
        query,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale=scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
    )
    
    return output


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # Dynamically get block_size to avoid hardcoding
        block_size = k_cache.shape[1] if k_cache.numel() > 0 else 16 

        if k_cache.numel() > 0 and v_cache.numel() > 0 and context.slot_mapping is not None:
            # Handle cases where k is (BS, Seq, ... ) or flattened
            if k.dim() == 4:
                B, N, num_kv_heads, head_dim = k.shape
                k_to_store = k.reshape(B * N, num_kv_heads, head_dim).contiguous()
                v_to_store = v.reshape(B * N, num_kv_heads, head_dim).contiguous()
            else:
                k_to_store = k.contiguous()
                v_to_store = v.contiguous()
            
            store_kvcache(k_to_store, v_to_store, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            cu_seqlens_q = context.cu_seqlens_q
            if context.block_tables is not None: 
                # [Branch 1] Prefill with prefix cache, requires block table lookup
                o = flash_attention_paged_prefill(
                    q, k_cache, v_cache, 
                    cu_seqlens_q, context.cu_seqlens_k, context.block_tables, 
                    self.scale, self.num_heads, self.num_kv_heads, self.head_dim, block_size
                )
            else:
                # [Branch 2] Ordinary variable-length Prefill (Warmup or first request), 
                # no table to lookup, directly compute continuous k, v
                o = flash_attention_prefill_ordinary(
                    q, k, v, 
                    cu_seqlens_q, context.max_seqlen_q, self.scale, self.num_heads, self.num_kv_heads, self.head_dim
                )
        else:
            # [Branch 3] Decode
            o = paged_attention_decode(
                q, k_cache, v_cache,
                context.block_tables, context.context_lens,
                self.scale, self.num_heads, self.num_kv_heads, self.head_dim, block_size
            )
        return o