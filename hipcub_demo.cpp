#include <stdio.h>
#include <hip/hip_runtime.h>

#define USE_HIPCUB 0

#if USE_HIPCUB == 1

  /* rocPRIM 4.4+ FAIL, HIP->CUDA WORKS! */
  #include "hipcub/hipcub.hpp"

#else 

  /* rocPRIM 4.3 WORKS! */
  #include "rocprim-4.3/include/rocprim/block/block_reduce.hpp"

  /* rocPRIM 4.4+ FAIL! */
  //#include "/opt/rocm/rocprim/include/rocprim/block/block_reduce.hpp"

#endif

/* Blocksize is divisible by the warp size */
#define BLOCKSIZE 64

/* CPU redution loop */
template <typename Lambda>
void parallel_reduce_cpu(const int loop_size, Lambda loop_body, int *sum) {
  // Evaluate the loop body
  for(int i = 0; i < loop_size; i++){
    loop_body(i, *sum);
  }
}

/* GPU redution kernel */
template <typename Lambda>
__global__ void reduction_kernel(Lambda loop_body, const int loop_size, int *sum)
{
  // Specialize BlockReduce for a 1D block of BLOCKSIZE threads of type int
  #if USE_HIPCUB == 1
    typedef hipcub::BlockReduce<int, BLOCKSIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
  #else
    using block_reduce_int = rocprim::block_reduce<int, BLOCKSIZE>;
    __shared__ block_reduce_int::storage_type temp_storage;
  #endif

  // Get thread index
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Check loop limits
  if (idx < loop_size) {
   
    // Local storage for the thread summation value
    int thread_sum = 0;
  
    // Evaluate the loop body, the summation value is stored in thread_sum
    loop_body(idx, thread_sum);
  
    // Compute the block-wide sum (aggregate) for the first thread of each block
    #if USE_HIPCUB == 1
      int aggregate = BlockReduce(temp_storage).Sum(thread_sum);
    #else
      int aggregate;
      block_reduce_int().reduce(thread_sum, aggregate, temp_storage, rocprim::plus<int>());
    #endif

    // The first thread of each block stores the block-wide aggregate to 'sum' using atomics
    if(threadIdx.x == 0) 
      atomicAdd(sum, aggregate);
  }
}

/* Wrapper for the GPU redution kernel */
template <typename Lambda>
void parallel_reduce_gpu(const uint loop_size, Lambda loop_body, int *sum) {

  // Set block and grid dimensions
  const uint blocksize = BLOCKSIZE;
  const uint gridsize = (loop_size - 1 + blocksize) / blocksize;

  // Create GPU buffer for the reduction variable
  int* d_buf;
  hipMalloc(&d_buf, sizeof(int));
  hipMemcpy(d_buf, sum, sizeof(int), hipMemcpyHostToDevice);

  // Launch the reduction kernel
  reduction_kernel<<<gridsize, blocksize>>>(loop_body, loop_size, d_buf);
  hipStreamSynchronize(0);
  
  // Copy reduction variable back to host from the GPU buffer
  hipMemcpy(sum, d_buf, sizeof(int), hipMemcpyDeviceToHost);
  hipFree(d_buf);
}


/* The main function */
int main()
{
  // Calculate the triangular number up to 'tn', ie, a sum of numbers from 0 to 'tn'
  const int tn = 1000;

  // Calculate the triangular number on the GPU and store it in sum_gpu
  int sum_gpu = 0;
  parallel_reduce_gpu(tn, [] __host__ __device__ (const int i, int &sum){
    int thread_idx = i;
    sum += thread_idx; 
  }, &sum_gpu);

  // Calculate the triangular number on the CPU and store it in sum_cpu
  int sum_cpu = 0;
  parallel_reduce_cpu(tn, [] __host__ __device__ (const int i, int &sum){
    int thread_idx = i;
    sum += thread_idx;
  }, &sum_cpu);

  // Check that the results match
  if(sum_gpu == sum_cpu)
    printf("The results calculated by GPU = %d and CPU = %d match!\n", sum_gpu, sum_cpu);
  else
    printf("The results calculated by GPU = %d and CPU = %d do not match!\n", sum_gpu, sum_cpu);

  return 0;
}
