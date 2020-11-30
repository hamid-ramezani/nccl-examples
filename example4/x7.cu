#include <stdio.h>
#include "cuda_runtime.h"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include<stdlib.h>


#define PACK_SIZE 8


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


template <int BITS>
__device__ void find_meta_parallel(float* input, unsigned char* meta, int num_elems) {
  unsigned int tid = threadIdx.x;
  unsigned int block_size = blockDim.x;
  float* meta_buf = (float*)meta;
  const int MAX_THREADS_PER_BLOCK = 1024;
  const int shared_size = MAX_THREADS_PER_BLOCK * 2;
  __shared__ float sdata[shared_size];
  meta_buf[0] = input[0];
  meta_buf[1] = input[0];
  unsigned int num_iters_per_bucket = (num_elems + block_size - 1) / block_size;
  for (int i = 0; i < num_iters_per_bucket; i++) {
    unsigned int idx = i * blockDim.x + tid;
    if (idx < num_elems) {
        sdata[tid] = input[idx];
        sdata[block_size + tid] = input[idx];
    }
    __syncthreads();

    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
      if (tid < s && idx + s < num_elems) {
        sdata[tid] = fmaxf(sdata[tid + s], sdata[tid]);
        sdata[block_size + tid] =
            fminf(sdata[block_size + tid + s], sdata[block_size + tid]);
      }
      __syncthreads();
    }

    if (tid == 0) {
        meta_buf[0] = fmaxf(meta_buf[0], sdata[tid]);
        meta_buf[1] = fminf(meta_buf[1], sdata[block_size + tid]);
    }
  }
  
  if (tid == 0) {
      const unsigned int divisor = (1 << BITS) - 1;
      meta_buf[0] = (meta_buf[0] - meta_buf[1]) / divisor;
  }
  __syncthreads();

  for (int i=0; i<2; ++i) { 
    printf("meta_buff[%d] = %f\n", i, meta_buf[i]);
  }

}

__global__ void kernel1(float* input, unsigned char* meta, size_t input_size, size_t meta_size) {
   for (int i=0; i<input_size; ++i) { 
     printf("input[%d] = %f\n", i, input[i]);
   }

   for (int i=0; i<meta_size; ++i) {
     printf("meta[%d] = %u\n", i, (unsigned int)meta[i]);
   }

   find_meta_parallel<3>(input, meta, (int)input_size);
}

int main() {
   size_t input_size = 8;
   size_t meta_size = 8;
   float* input;
   float** input_ptr = &input;
   CUDACHECK(cudaMalloc((void**)input_ptr, input_size * sizeof(float)));

   thrust::device_ptr<float> dev_ptr1(input);
   for (int i=0; i<input_size; i++) {
     thrust::fill(dev_ptr1+i, dev_ptr1+i+1, static_cast<float>(i+1));
   }

   unsigned char* meta;
   unsigned char** meta_ptr = &meta;
   CUDACHECK(cudaMalloc((void**)meta_ptr, meta_size * sizeof(unsigned char)));
   CUDACHECK(cudaMemset((void*)meta, (unsigned char)97, meta_size * sizeof(unsigned char)));

   kernel1<<<1,1>>>(input, meta, input_size, meta_size);

   unsigned char* meta_host = (unsigned char*)malloc(2 * sizeof(unsigned char)); 
   CUDACHECK(cudaMemcpy(meta_host, meta, 2*sizeof(unsigned char), cudaMemcpyDeviceToHost));

   for (int i=0; i<2; ++i) {
     printf("updated meta[%d] = %f\n", i, (float)meta_host[i]);
   }

   CUDACHECK(cudaPeekAtLastError());
   CUDACHECK(cudaDeviceSynchronize());

   return 0;
}
