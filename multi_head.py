##Below is the code for Multi-Head Attention (CUDA Powered) 
#----------------------------------------------------------

import numpy as np
from attention_gpu import self_attention_cuda, gpu_matmul

#Multi-head wrapper 
def multi_head_attention_cuda (X, W_Q, W_K, W_V, W_O, num_heads):
    """
    Parameters:
        X: (seq_length, d_model)
        W_Q: (d_model, d_k * num_heads) 
        W_K: (d_model, d_k * num_heads)
        W_V: (d_model, d_v * num_heads)
        W_O: (d_v * num_heads, d_model)
        num_heads: number of attention heads

    Returns:
    final_output: (seq_length, d_model)
    all_attention
    
    """
    seq_length, d_model = X.shape 

    #figuring how wide each head is 
    d_k = W_Q.shape[1] // num_heads 
    d_v = W_V.shape[1] // num_heads 

    outputs = []
    all_attention_weights = []

    #looping through each attention head
    for i in range (num_heads):

        #slicing the weights so each attention head has its own full weight matrices. 
        W_Q_i = W_Q[:, i*d_k:(i+1)*d_k]
        W_K_i = W_K[:, i*d_k:(i+1)*d_k]
        W_V_i = W_V[:, i*d_v:(i+1)*d_v] 

        #each head runs self_attention_cuda to focus on different parts of the input. 
        out_i, attn_i = self_attention_cuda(X, W_Q_i, W_K_i, W_V_i)

        #stitching the attention heads together 
        outputs.append(out_i)
        all_attention_weights.append(attn_i)
        
    #Concatenate all heads: (seq_len, d_v * num_heads) 
    concat = np.concatenate(outputs, axis = -1).astype(np.float32)

    #Final linear layer: after the heads are concatenated together, it needs to be projected back to its original size using matmul 
    final_output = gpu_matmul(concat, W_O)

    return final_output, all_attention_weights
