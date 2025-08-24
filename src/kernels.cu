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

template <typename T>
__global__ void
flashAttentionKernel(const T *__restrict__ Q, const T *__restrict__ K,
                              const T *__restrict__ V, T *O, 
                              T *global_scores, // 全局内存用于存储中间scores
                              int batch_size, int target_seq_len, 
                              int src_seq_len, int query_heads,
                              int kv_heads, int head_dim, bool is_causal) {
  int b = blockIdx.x;  // batch
  int qh = blockIdx.y; // query head
  
  // Sequence position calculation:
  // blockIdx.z: which segment of the sequence this block handles
  int t_start = blockIdx.z * blockDim.x;
  int t = t_start + threadIdx.x;
  
  // Boundary check: exit if out of valid range
  if (b >= batch_size || t >= target_seq_len || qh >= query_heads)
    return;
  
  // Grouped-query attention (GQA) factor
  int group_size = query_heads / kv_heads;
  
  // Calculate memory offset for current Q vector
  int q_offset = ((b * target_seq_len + t) * query_heads + qh) * head_dim;

  // Attention scaling factor (1/sqrt(d))
  T scale_factor = static_cast<T>(1.0 / sqrtf(static_cast<float>(head_dim)));

  int scores_offset = ((b * query_heads + qh) * target_seq_len + t) * src_seq_len;

  // Phase 1: Compute attention scores for Q[t] x K[s] for all s
  for (int s = 0; s < src_seq_len; ++s) {
     // Causal masking: set future positions to -inf
    if (is_causal && s > t) {
      global_scores[scores_offset + s] = static_cast<T>(-1e9);
      continue;
    }
    
    // For GQA: map query head to key head
    int kh = qh / group_size;
    int k_offset = ((b * src_seq_len + s) * kv_heads + kh) * head_dim;

    T dot = static_cast<T>(0);
    for (int d = 0; d < head_dim; d++) {
      dot += Q[q_offset + d] * K[k_offset + d];
    }
    // Dot product: Q[t] • K[s]
    global_scores[scores_offset + s] = dot * scale_factor; // Scaled score
  }

  __syncthreads();

  // Phase 2: Softmax normalization (per thread)
  // Step 1: Find row-wise max for numerical stability
  T max_val = static_cast<T>(-1e9);
  for (int s = 0; s < src_seq_len; ++s) {
    max_val = max(max_val, global_scores[scores_offset + s]);
  }
  
  // Step 2: Compute exponentials and sum
  T sum_val = static_cast<T>(0);
  for (int s = 0; s < src_seq_len; ++s) {
    T exp_val = expf(global_scores[scores_offset + s] - max_val);
    global_scores[scores_offset + s] = exp_val;
    sum_val += exp_val;
  }

  // Step 3: Normalize by sum to get probabilities
  for (int s = 0; s < src_seq_len; s++) {
    global_scores[scores_offset + s] /= sum_val;
  }

  __syncthreads();

  // Phase 3: Weighted sum of V vectors (Attention output)
  int o_offset = ((b * target_seq_len + t) * query_heads + qh) * head_dim;
  for (int d = 0; d < head_dim; ++d) {
    T val = static_cast<T>(0);
    for (int s = 0; s < src_seq_len; ++s) {
      // GQA: map query head to value head (same as key head)
      int kh = qh / group_size;
      int v_offset = ((b * src_seq_len + s) * kv_heads + kh) * head_dim;
      val += global_scores[scores_offset + s] * V[v_offset + d];
    }
    O[o_offset + d] = val;
  }
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

  // Validate Grouped-Query Attention constraint
  if (query_heads % kv_heads != 0) {
    throw std::invalid_argument("query_heads must be divisible by kv_heads");
  }
  
  // Calculate expected tensor sizes (in elements)
  size_t size_q = (size_t)batch_size * target_seq_len * query_heads * head_dim;
  size_t size_k = (size_t)batch_size * src_seq_len * kv_heads * head_dim;
  size_t size_v = size_k;
  size_t size_o = size_q;
 
  // Validate input/output vector sizes
  if (h_q.size() != size_q || h_k.size() != size_k || h_v.size() != size_v) {
    throw std::invalid_argument("Input vector sizes do not match expected dimensions");
  }
  
  // Device memory allocation
  T *d_q, *d_k, *d_v, *d_o;
  CUDA_CHECK(cudaMalloc(&d_q, size_q * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_k, size_k * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_v, size_v * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_o, size_o * sizeof(T)));

  // Copy input data to device
  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), size_q * sizeof(T), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), size_k * sizeof(T), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), size_v * sizeof(T), cudaMemcpyHostToDevice));

  // Split sequence dimension into blocks
  size_t blocks_per_sequence = (target_seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 grid(batch_size, query_heads, blocks_per_sequence);
  dim3 block(BLOCK_SIZE);
  
  // Allocate global memory to store intermediate scores
  size_t scores_size = (size_t)batch_size * query_heads * target_seq_len * src_seq_len;
  T *d_scores;
  CUDA_CHECK(cudaMalloc(&d_scores, scores_size * sizeof(T)));
  
  // Launch kernel with global memory
  flashAttentionKernel<T><<<grid, block>>>(
      d_q, d_k, d_v, d_o, d_scores, batch_size, target_seq_len, src_seq_len, 
      query_heads, kv_heads, head_dim, is_causal);
  
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Copy results back to host
  CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, size_o * sizeof(T), cudaMemcpyDeviceToHost));

  // Cleanup device memory
  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_o));
  CUDA_CHECK(cudaFree(d_scores));                   
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
