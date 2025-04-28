
<h3 align="center">CUDA-Accelerated Transformer Encoder Block</h3>

  <p align="center">
    Implementation of a Transformer encoder block with CUDA acceleration for enhanced performance.
    <br />
     <a href="https://github.com/krupadav3/encoder-block-in-cuda">github.com/krupadav3/encoder-block-in-cuda</a>
  </p>
</div>

## Table of Contents

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#key-features">Key Features</a></li>
      </ul>
    </li>
    <li><a href="#architecture">Architecture</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project

This project implements a single Transformer encoder block using CUDA for significant performance gains. The encoder block consists of multi-head attention, add & norm layers, and a feed-forward network.  Each component is optimized to leverage the parallel processing capabilities of GPUs, resulting in faster training and inference times.

### Key Features

- **CUDA Acceleration:** Utilizes CUDA kernels for matrix multiplication, softmax, and layer normalization, providing substantial speedups compared to CPU implementations.
- **Multi-Head Attention:** Implements multi-head attention mechanism to capture different relationships within the input sequence.
- **Add & Norm:** Includes residual connections and layer normalization for improved training stability and performance.
- **Feed-Forward Network:**  A feed-forward network (FFN) implemented with CUDA to transform each token independently.
- **Positional Encoding:** Applies sinusoidal positional embeddings to the input to provide information about the position of tokens in the sequence.

## Architecture

The project is structured as follows:

- **`add_norm.py`:** Implements the Add & Norm layer with a CUDA-accelerated LayerNorm kernel.
- **`attention_gpu.py`:** Contains CUDA kernels for matrix multiplication and softmax, used in the self-attention mechanism.
- **`encoder_block.py`:** Defines the `encoder_block` function, which combines the multi-head attention, add & norm, and feed-forward network.
- **`ffn_cuda.py`:** Implements the feed-forward network with CUDA acceleration.
- **`multi_head.py`:** Implements the multi-head attention mechanism, utilizing the CUDA-accelerated self-attention.
- **`positional_encoding.py`:** Implements sinusoidal positional encoding.

The key technologies used are:

- **Numba:** Used to compile Python code to CUDA kernels for GPU execution.
- **CUDA:** NVIDIA's parallel computing platform and programming model.
- **NumPy:** Used for numerical operations and array manipulation.

## Getting Started

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Numba
  ```sh
  pip install numba
  ```
- NumPy
  ```sh
  pip install numpy
  ```

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/krupadav3/encoder-block-in-cuda.git
   ```
2. Navigate to the project directory:
   ```sh
   cd encoder-block-in-cuda
   ```
3.  No further installation steps are required. You can directly run the python files, such as the optional test in `add_norm.py` or the example in `ffn_cuda.py`.
