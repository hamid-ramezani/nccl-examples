#include <stdio.h>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


template <int BITS>
__device__ void find_meta_seq(const float* input, unsigned char* meta, int num_elem, int bucket_size) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  const unsigned int divisor = (1 << BITS) - 1;
  float* meta_buf = (float*)meta;
  for (int i = index; i < (num_elem + bucket_size - 1) / bucket_size; i += stride) {
    float mmin = input[i * bucket_size];
    float mmax = input[i * bucket_size];
    for (int j = i * bucket_size + 1; j < fminf((i + 1) * bucket_size, num_elem);
         j++) {
      mmin = fminf(mmin, input[j]);
      mmax = fmaxf(mmax, input[j]);
    }
    meta_buf[2 * i] = static_cast<float>((mmax - mmin) / divisor);
    meta_buf[2 * i + 1] = mmin;
  }
  
  float a = static_cast<float>(num_elem)/static_cast<float>(bucket_size);
  size_t meta_buff_size = 2*ceil(a);
  for (int i=0; i<meta_buff_size; ++i) { 
    printf("meta_buff[%d] = %f\n", i, meta_buf[i]);
  }
}


__global__ void kernel1(const float* input, unsigned char* meta, size_t input_size, size_t meta_size) {
   for (int i=0; i<input_size; ++i) { 
     printf("input[%d] = %f\n", i, input[i]);
   }

   for (int i=0; i<meta_size; ++i) {
     printf("meta[%d] = %f\n", i, static_cast<float>(meta[i]));
   }

   find_meta_seq<3>(input, meta, input_size, 2);
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
   CUDACHECK(cudaMemset((void*)meta, 97, meta_size * sizeof(unsigned char)));

   kernel1<<<1,1>>>((const float*)input, meta, input_size, meta_size);

   CUDACHECK(cudaPeekAtLastError());
   CUDACHECK(cudaDeviceSynchronize());

   //unsigned char* meta_host = (unsigned char*)malloc(8 * sizeof(unsigned char));
   //CUDACHECK(cudaMemcpy(meta_host, meta, sizeof(meta)*sizeof(unsigned char), cudaMemcpyDeviceToHost));

   return 0;
}
