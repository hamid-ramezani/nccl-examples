#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


int main(int argc, char* argv[])
{
  ncclComm_t comms[2];
  //ncclComm_t comms[4];

  //managing 4 devices
  //int nDev = 1;
  int nDev = 2;
  //int nDev = 4;

  //int size = 32*1024*1024;
  //int size = 128*32*32;
  int size = 8192;
  //int size = 8;


  //int devs[1] = { 0 };
  int devs[2] = { 0, 1 };
  //int devs[4] = { 0, 1, 2, 3 };
  //size_t  heapSize = 1024 * 1024 * 1024;
  //cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);

  //allocating and initializing device buffers
  //int8_t** sendbuff = (int8_t**)malloc(nDev * sizeof(int8_t*));
  //int8_t** recvbuff = (int8_t**)malloc(nDev * sizeof(int8_t*));
  //int8_t** tempbuff = (int8_t**)malloc(nDev * sizeof(int8_t*));
  
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  //int8_t** recvbuff = (int8_t**)malloc(nDev * sizeof(int8_t*));
  //float** tempbuff = (float**)malloc(nDev * sizeof(float*));
  
  cudaStream_t* s = (cudaStream_t*)malloc(nDev * sizeof(cudaStream_t));
  
  //int8_t* h_sendbuff = (int8_t*)malloc(size * sizeof(int8_t));
  //int8_t* h_recvbuff = (int8_t*)malloc(size * sizeof(int8_t));
  
  float* h_sendbuff = (float*)malloc(size * sizeof(float));
  float* h_recvbuff = (float*)malloc(size * sizeof(float));
  //int8_t* h_recvbuff = (int8_t*)malloc(size * sizeof(int8_t));
  
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    //CUDACHECK(cudaMalloc((void**)sendbuff + i, size * sizeof(int8_t)));
    //CUDACHECK(cudaMalloc((void**)recvbuff + i, size * sizeof(int8_t)));
    //CUDACHECK(cudaMemset(sendbuff[i], 13, size * sizeof(int8_t)));
    //CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(int8_t)));

    //CUDACHECK(cudaMalloc((void**)tempbuff + i, size * sizeof(int8_t)));
    //CUDACHECK(cudaMemset(tempbuff[i], 0, size * sizeof(int8_t)));
  
    CUDACHECK(cudaMalloc((void**)sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc((void**)recvbuff + i, size * sizeof(float)));
    //CUDACHECK(cudaMalloc((void**)recvbuff + i, size * sizeof(int8_t)));

    //float fill_value1 = 0.014;
    //float fill_value2 = 0.011;
    //float fill_value3 = 0.012;
    //float fill_value4 = 0.013;
    //float fill_value5 = 0.015;
    //float fill_value6 = 0.016;
    //float fill_value7 = 0.020;
    //float fill_value8 = 0.017;
    //thrust::device_ptr<float> dev_ptr1(sendbuff[i]);
    //thrust::fill(dev_ptr1, dev_ptr1 + 1, fill_value1);
    //thrust::fill(dev_ptr1 + 1, dev_ptr1 + 2, fill_value2);
    //thrust::fill(dev_ptr1 + 2, dev_ptr1 + 3, fill_value3);
    //thrust::fill(dev_ptr1 + 3, dev_ptr1 + 4, fill_value4);
    //thrust::fill(dev_ptr1 + 4, dev_ptr1 + 5, fill_value5);
    //thrust::fill(dev_ptr1 + 5, dev_ptr1 + 6, fill_value6);
    //thrust::fill(dev_ptr1 + 6, dev_ptr1 + 7, fill_value7);
    //thrust::fill(dev_ptr1 + 7, dev_ptr1 + 8, fill_value8);

    thrust::device_ptr<float> dev_ptr1(sendbuff[i]);
    thrust::fill(dev_ptr1, dev_ptr1 + size, 2.4);

    thrust::device_ptr<float> dev_ptr2(recvbuff[i]);
    thrust::fill(dev_ptr2, dev_ptr2 + size, 0.0);

    //CUDACHECK(cudaMemset(recvbuff[i], 0., size * sizeof(int8_t)));


    //CUDACHECK(cudaMemset(sendbuff[i], 2., size * sizeof(float)));
    //CUDACHECK(cudaMemset(recvbuff[i], 0., size * sizeof(float)));

    //CUDACHECK(cudaMalloc((void**)tempbuff + i, size * sizeof(float)));
    //CUDACHECK(cudaMemset(tempbuff[i], 0, size * sizeof(float)));
  
    CUDACHECK(cudaStreamCreate(s+i));
    CUDACHECK(cudaDeviceSynchronize());
  }

   //CUDACHECK(cudaMemcpy(h_sendbuff,sendbuff[0],size * sizeof(int8_t),cudaMemcpyDeviceToHost));
   CUDACHECK(cudaMemcpy(h_sendbuff,sendbuff[0],size * sizeof(float),cudaMemcpyDeviceToHost));
   CUDACHECK(cudaDeviceSynchronize());
   
  //initializing NCCL
  //calling NCCL communication API. Group API is required when using
  //multiple devices per thread
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i){
    //NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclInt8 , ncclSum, comms[i], s[i]));
    NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum, comms[i], s[i]));
    //NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], (void*)tempbuff[i], size, ncclInt8 , ncclSum, comms[i], s[i]));
    //NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], (void*)tempbuff[i], size, ncclFloat , ncclSum, comms[i], s[i]));
  }
  NCCLCHECK(ncclGroupEnd());


  //CUDACHECK(cudaMemcpy(h_recvbuff,recvbuff[0],size * sizeof(int8_t),cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(h_recvbuff, recvbuff[0], size * sizeof(float),cudaMemcpyDeviceToHost));
  CUDACHECK(cudaDeviceSynchronize());

  //for (int i = 0; i< size; ++i) {
  //  printf("%f\n",h_sendbuff[i]);
  //}
 
  //for (int i = 0; i< size; ++i) {
  //  printf("%f\n",h_recvbuff[i]);
  //}

  int count = 0;
  for(int i=0; i<size; ++i) {
    if(abs(h_recvbuff[i] - 4.8) > 0.0001f){
       count++;
       printf("h_recvbuff[%d] = %f \n", i, h_recvbuff[i]);
    }
  }
  printf("count = %d \n", count);

 //synchronizing on CUDA streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }

  //free device buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
    //CUDACHECK(cudaFree(tempbuff[i]));
  }


  //finalizing NCCL
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);


  printf("Success \n");
  return 0;
}
