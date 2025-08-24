#include <iostream>
#include <vector>

#include "../tester/utils.h"

#define BLOCK_SIZE 256

/* Set array values kernel: used for padding*/
template <typename T> __global__ void set_val(T *array, int size, T val) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    array[idx] = val;
  }
}

/* Device function for compare-and-swap operation */
template <typename T>
__device__ void compare_and_swap(T &a, T &b, bool direction) {
  bool flag = a > b;
  if (flag == direction) {
    T tmp = a;
    a = b;
    b = tmp;
  }
}

/* Core bitonic sort kernel */
template <typename T> __global__ void bitonic_sort(T *array, int i, int j) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int pair = idx ^ j;
  bool direction = ((idx / i) % 2) == 0;
  if (idx < pair) {
    compare_and_swap(array[idx], array[pair], direction);
  }
}

/**
 * @brief Find the k-th largest element in a vector using CUDA.
 * 
 * @tparam T Type of elements in the input vector (should support `int` and `float`).
 * @param h_input Host-side input vector.
 * @param k 1-based index of the element to find (e.g., `k=1` returns the largest element).
 * @return T The k-th largest element in `h_input`.

 * @note Must use CUDA kernels for all compute-intensive steps; no significant CPU allowed.
 * @note Library functions that can directly complete a significant part of the work are NOT allowed. 
 * @note For invalid cases, return T(-100).
 * @note Handles device memory management (allocate/copy/free) internally. Errors should be thrown.
 */
template <typename T>
T kthLargest(const std::vector<T>& h_input, size_t k) {
  // 1. Validate input parameters
  if (h_input.empty() || k < 1 || k > h_input.size()) {
    return T(-100);
  }

  // 2. Calculate padded size (next power-of-two)
  size_t n = h_input.size();
  double log_val = std::log2(static_cast<double>(n));
  size_t log_ceil = static_cast<size_t>(std::ceil(log_val));
  size_t padded_n = static_cast<size_t>(1) << log_ceil;

  size_t num_pads = padded_n - n;

  // 3. Set padding value based on data type
  T pad_val;
  if constexpr (std::is_floating_point_v<T>) {
    pad_val = -std::numeric_limits<T>::infinity();
  } else {
    pad_val = std::numeric_limits<T>::lowest();
  }

  // 4. Device memory operations
  T *d_padded = nullptr;
  CUDA_CHECK(cudaMalloc(&d_padded, padded_n * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d_padded, h_input.data(), n * sizeof(T),
                        cudaMemcpyHostToDevice));

  // 5. Pad extra positions with min-value
  dim3 block(BLOCK_SIZE);
  dim3 grid((num_pads - 1) / BLOCK_SIZE + 1);
  if (num_pads > 0) {
    set_val<<<block, grid>>>(d_padded + n, num_pads, pad_val);
  }

  // 6. Execute bitonic sort
  grid = (padded_n - 1) / BLOCK_SIZE + 1;
  for (int i = 2; i <= padded_n; i = i * 2) {
    for (int j = i / 2; j > 0; j = j / 2) {
      bitonic_sort<<<grid, block>>>(d_padded, i, j);
    }
  }

  // 7. Retrieve k-th largest element (at index padded_n - k in ascending order)
  size_t index = padded_n - k;
  T result;
  CUDA_CHECK(
      cudaMemcpy(&result, d_padded + index, sizeof(T), cudaMemcpyDeviceToHost));

  // 8. Cleanup device memory
  CUDA_CHECK(cudaFree(d_padded));
  return result;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int kthLargest<int>(const std::vector<int>&, size_t);
template float kthLargest<float>(const std::vector<float>&, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
