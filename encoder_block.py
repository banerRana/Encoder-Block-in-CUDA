import numpy as np
from attention_gpu import self_attention_cuda
from multi_head import multi_head_attention_cuda
from positional_encoding import apply_positional_encoding
from add_norm import add_and_norm
from ffn_cuda import ffn_cuda

def encoder_block (X, W_Q, W_K, W_V, W_O, W1, b1, W2, b2, num_heads):
    
    """
    Runs a single Transformer encoder block.

    Args:
        X: Input embeddings (seq_len, d_model)
        W_Q, W_K, W_V: (d_model, d_k * num_heads)
        W_O = (d_v * num_heads, d_model)
        W1: First FFN weight matrix (d_model, hidden_dim)
        b1: First FFN bias vector (hidden_dim)
        W2: Second FFN weight matrix (hidden_dim, d_model)
        b2: Second FFN bias vector (d_model)
        num_heads: number of attentino heads

    Returns:
        Output of the encoder block (seq_len, d_model)
    """
    # Step 1: Apply multi-head attention
    attn_output, _ = multi_head_attention_cuda(X, W_Q, W_K, W_V, W_O, num_heads)

    # Step 2: Add & Norm (residual connection after attention)
    norm1 = add_and_norm(X, attn_output)

    # Step 3: Feed-forward network
    ffn_out = ffn_cuda(norm1, W1, b1, W2, b2)

    # Step 4: Add & Norm (residual connection after FFN)
    final_output = add_and_norm(norm1, ffn_out)

    return final_output

