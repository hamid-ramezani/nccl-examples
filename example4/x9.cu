#include <stdio.h>
#include "cuda_runtime.h"
#include <stdint.h>
//#include <thrust/device_ptr.h>
//#include <thrust/fill.h>

#define PACK_SIZE 8
#define EPS 1e-10

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


template <int BITS>
__device__ void find_meta_seq(const float* input, float* meta, int num_elem, int bucket_size) {
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
  
  //float a = static_cast<float>(num_elem)/static_cast<float>(bucket_size);
  //size_t meta_buff_size = 2*ceil(a);
  //for (int i=0; i<meta_buff_size; ++i) { 
  //  printf("meta_buff[%d] = %f\n", i, meta_buf[i]);
  //}
}



inline __device__ unsigned char
MaxMinEncodeValue(float input, float* meta_info, float rand) {
  float* maxmin = ((float*)meta_info);
  if (maxmin[0] < EPS) {
    return 0;
  }
  float min = maxmin[1];
  float unit = maxmin[0];
  float d = (input - min) / unit + rand;
  unsigned char level = floor(d);
  return level;
}


template <int BITS>
__device__ void CompressBucket(float* input, unsigned char* output, float* meta_info, int num_elems) {
  using uint64_t = unsigned long long int;
  unsigned int tid = threadIdx.x;
  unsigned int num_threads = blockDim.x;
  float rand;
  int num_char = (BITS * num_elems + PACK_SIZE - 1) / PACK_SIZE;
  for (unsigned int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE; i += num_threads) {
      uint64_t value = 0;
      for (unsigned int j = 0; j < PACK_SIZE && i * PACK_SIZE + j < num_elems; j++) {
        int idx = i * PACK_SIZE + j;
        rand = 0.5;
        //rand = GetRand(state);
        uint64_t encoded = MaxMinEncodeValue(input[idx], meta_info, rand);
        value += (encoded << (j * BITS));
      }
      for (unsigned int j = 0; j < BITS && i * BITS + j < num_char; j++) {
        output[i * BITS + j] = value >> (PACK_SIZE * j) & 0xFF;
      }
  }
  //for (int i=0; i<num_char; ++i) {
  //  printf("updated output[%d] = %u \n", i, (unsigned int)output[i]);
  //}
}


template <int BITS>
__device__ void quantize(float* input_data, unsigned char* output_data, int num_elems, int bucket_size) {
  unsigned int num_blocks = gridDim.x;
  //unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int bid = blockIdx.x;
  unsigned int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  unsigned int cur_bucket_size;

  float* meta = (float*)output_data;
  unsigned char* output;
  const int meta_multiplier = 2;
  output = output_data + meta_multiplier * sizeof(float) * num_buckets;

  unsigned int compressed_size = (bucket_size * BITS + PACK_SIZE - 1) / PACK_SIZE;

  float* input = (float*)input_data;
  find_meta_seq<BITS>(input, meta, num_elems, bucket_size);
  //for (int bucket_id = bid; bucket_id < num_buckets; bucket_id += num_blocks) {
  //  cur_bucket_size = umin(bucket_size, num_elems - bucket_id * bucket_size);
  //  find_meta_parallel<BITS>(
  //      input + bucket_size * bucket_id,
  //      (meta + meta_multiplier * bucket_id), cur_bucket_size);
  //}
  for (int bucket_id = bid; bucket_id < num_buckets; bucket_id += num_blocks) {
    cur_bucket_size = umin(bucket_size, num_elems - bucket_id * bucket_size);
    CompressBucket<BITS>(
        input + bucket_size * bucket_id, output + compressed_size * bucket_id,
        (meta + meta_multiplier * bucket_id),
        cur_bucket_size);
  }

  printf("The output of quantization is as follows: \n");
  for (int i=0; i<num_elems; ++i) {
    printf("quantized array[%d] = %u \n", i, output_data[i]);
  }
}



inline __device__ float MaxMinDecodeValue(unsigned char input, float* meta_info, unsigned int idx, int bucket_size) {
  int bucket_no = idx / bucket_size;
  float* maxmin = ((float*)meta_info) + 2 * bucket_no;
  float min = maxmin[1];
  float unit = maxmin[0];
  return min + input * unit;
}

template <bool ADD, int BITS>
__device__ void dequantize(unsigned char* input_data, float* output, int num_elems, int bucket_size) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;

  unsigned int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  float* meta_info = (float*)input_data;

  //printf("meta_info is as follows: \n");
  //for(int i=0; i<8; i++) { 
  //  printf("meta_info(%d) = %f\n",i, meta_info[i]);
  //}
  
  unsigned char* input; 
  const int meta_multiplier = 2;
  input = input_data + meta_multiplier * sizeof(float) * num_buckets;
  
  int num_char = (BITS * num_elems + PACK_SIZE - 1) / PACK_SIZE;
  unsigned int divisor = 1 << BITS;
  for (unsigned int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE; i += stride) {
    uint64_t value = 0;
    for (int j = 0; j < BITS && i * BITS + j < num_char; j++) {
      value |= ((uint64_t)input[i * BITS + j]) << (j * PACK_SIZE);
    }
    for (int j = 0; j < PACK_SIZE && i * PACK_SIZE + j < num_elems; j++) {
      unsigned char encoded_value = (value >> (j * BITS)) & (divisor - 1);
      float d = MaxMinDecodeValue(encoded_value, meta_info, i * PACK_SIZE + j, bucket_size);
      if (ADD) {
        output[i * PACK_SIZE + j] = output[i * PACK_SIZE + j] + d;
      } else {
        output[i * PACK_SIZE + j] = d;
      }
    }
  }

  printf("The output of dequantization is as follows: \n");
  for (int i=0; i<num_elems; ++i) {
    printf("dequantized array[%d] = %f\n", i, output[i]);
  }
}


__global__ void kernel(float* input, unsigned char* output, float* dequantized_buff, int input_size, int output_size) {
   for (int i=0; i<input_size; ++i) {
     printf("input[%d] = %f\n", i, input[i]);
   }
   quantize<8>(input, output, input_size, 2);
   dequantize<true,8>(output, dequantized_buff, input_size, 2);
}

__global__ void initArray(float* array, int size, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i=idx; i<size; i += gridDim.x * blockDim.x) {
        array[i]=val;
    }
}

int main() {
   int num_elems = 8;
   int BITS = 8;

   float* input;
   float** input_ptr = &input;
   CUDACHECK(cudaMalloc((void**)input_ptr, num_elems * sizeof(float)));
   
   for (int i=0; i<num_elems; i++) {
     initArray<<<1,1>>>(input+i, 1, static_cast<float>((i+1)*(i+1)));
   }

   unsigned char* quantized_buff;
   CUDACHECK(cudaMalloc((void**)(&quantized_buff), 8*sizeof(float) + (BITS*num_elems/8)*sizeof(unsigned char)));

   float* dequantized_buff;
   float** dequantized_buff_ptr = &dequantized_buff;
   CUDACHECK(cudaMalloc((void**)dequantized_buff_ptr, num_elems*sizeof(float)));
   CUDACHECK(cudaMemset((void*)dequantized_buff, (float)0, num_elems * sizeof(float)));

   kernel<<<1,1>>>(input, quantized_buff, dequantized_buff, num_elems, num_elems);
   CUDACHECK(cudaPeekAtLastError());
   CUDACHECK(cudaDeviceSynchronize());
   return 0;
}
