

from math import sqrt
import numpy as np
from numba import cuda, float32

#--------LayerNorm kernel (row-wise) ---------
@cuda.jit
def layernorm_kernel (X, mean, var, gamma, beta, output, eps):

    #each thread on the GPU gets a unique row number to work on
    row = cuda.grid(1)
    if row < X.shape[0]:
        
        #for each column in the row 
        for j in range(X.shape[1]):

            #normalization formula (subtracting the row's mean so it's centered around 0 & then dividing by the standard deviation (square root of variance). eps avoids division by 0. 
            
            norm = (X[row, j] - mean[row, 0]) / sqrt(var[row, 0] + eps)

            #this is called "scaling and shifting" where gamma controls the scale & beta moves it up/down. It lets the model learn how much it wants to undo the normalization later 
            
            output[row, j] = gamma[j] * norm + beta[j]

#--------Wrapper function-----------
#this is the function that's called from Python and runs everything to make sure LayerNorm happends on the GPU

def layernorm_cuda (X, eps=1e-5):
    
    #converting inputs into float32
    X = X.astype (np.float32)
    m, d = X.shape

    #calculating the mean & variance for each row (token) - these are used for normalization.
    #note how gamma = 1 & beta = 0 for now (clean LayerNorm) which become learnable parameters later
    mean = np.mean(X, axis=1, keepdims=True).astype(np.float32)
    var = np.var(X, axis=1, keepdims=True).astype(np.float32)
    gamma = np.ones(d, dtype=np.float32)
    beta = np.zeros(d, dtype=np.float32)

    #preparing an empty array to store the results
    out = np.zeros_like(X)

    #copying everyting to the GPU 
    dX = cuda.to_device(X)
    dmean = cuda.to_device(mean)
    dvar = cuda.to_device(var)
    dgamma = cuda.to_device(gamma)
    dbeta = cuda.to_device(beta)
    dOut = cuda.to_device(out)

    
    threads_per_block = 32 #32 is picked as the safe default for the number of threads
    blocks_per_grid = (m + threads_per_block - 1) // threads_per_block #formula to calculate the number of blocks

    #launching the GPU kernel with all the needed inputs
    layernorm_kernel[blocks_per_grid, threads_per_block](dX, dmean, dvar, dgamma, dbeta, dOut, eps)

    #getting the results back from the GPU to the CPU2
    return dOut.copy_to_host()

#--------------Add & Norm Wrapper -------------
def add_and_norm (X, sublayer_output):

    #this adds the original input(X) to the sublayer output (residual connection)
    residual = X + sublayer_output
    return layernorm_cuda(residual)
    
# Optional test
if __name__ == "__main__":
    dummy = np.random.rand(4, 8).astype(np.float32)
    altered = dummy + np.random.rand(4, 8).astype(np.float32)
    result = add_and_norm(dummy, altered)
    print("Result after Add & Norm:")
    print(result)
    


