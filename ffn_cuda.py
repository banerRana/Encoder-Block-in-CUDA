#the FFN is the part of the encoder that transforms each token independently. It's where you give each word its own mini MLP (multi-layer perception)
#Input -> Linear -> ReLU --> Linear --> Output
#MLP performs multiple layers of nonlinear transformations on the input & the output serves as attention weights




import numpy as np
from numba import cuda 
from attention_gpu import gpu_matmul

#------ kernel for ReLU function -------
@cuda.jit
def relu_kernel (X, out):
    
    #gives each thread a (row, col) index so 1 thread = 1 cell in the matrix
    row, col = cuda.grid(2)

    #checking if the thread is within bounds of the matrix
    if row < X.shape[0] and col < X.shape[1]:
        val = X[row, col]

        #applying Relu by storing the results in out (if the number is negative in the matrix, then turn it zero)
        out [row, col] = max (0.0, val)

#-------Feed Forward Network wraper--------
def ffn_cuda (X, W1, b1, W2, b2):
    """

    Parameters:
        X: (seq_len, d_model)
        W1: (d_model, d_ff)
        b1: (d_ff)
        W2: (d_ff, d_model)
        b2: (d_model)

    Returns:
        Output: (seq_len, d_model)

    """

    #Step 1: Linear Projection 1 (X @W1 + b1)
    intermediate = gpu_matmul(X, W1) + b1

    #Step 2: ReLU Activation
    seq_len, d_ff = intermediate.shape
    relu_out = np.zeros_like(intermediate)

    threads = (16, 16)
    blocks = ((seq_len + threads[0] - 1) // threads[0], (d_ff + threads [1] - 1) // threads[1])

    d_intermediate = cuda.to_device(intermediate)
    d_relu_out = cuda.to_device(relu_out)
    relu_kernel[blocks, threads](d_intermediate, d_relu_out)
    relu_out = d_relu_out.copy_to_host()

    #Step 3: Linear Projection 2 
    output = gpu_matmul(relu_out, W2) + b2
    return output

if __name__ == "__main__":
    np.random.seed(0)

    seq_len = 4
    d_model = 8
    d_ff = 32

    X = np.random.rand(seq_len, d_model).astype(np.float32)
    W1 = np.random.rand(d_model, d_ff).astype(np.float32)
    b1 = np.random.rand(d_ff).astype(np.float32)
    W2 = np.random.rand(d_ff, d_model).astype(np.float32)
    b2 = np.random.rand(d_model).astype(np.float32)

    out = ffn_cuda(X, W1, b1, W2, b2)
    print("FFN Output:")
    print(out)

    

    
    






    

    