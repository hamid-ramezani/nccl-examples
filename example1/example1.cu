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

  ncclComm_t comms[4];

  //managing 4 devices
  int nDev = 4;

  //int size = 256*1024*1024;
  //int size = 32*32*32;
  int size = 8;

  int devs[4] = { 0, 1, 2, 3 };
  //int devs[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  //size_t  heapSize = 1024 * 1024 * 1024;
  //cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);


  //allocating and initializing device buffers
  //int8_t** sendbuff = (int8_t**)malloc(nDev * sizeof(int8_t*));
  //int8_t** recvbuff = (int8_t**)malloc(nDev * sizeof(int8_t*));
  //int8_t** tempbuff = (int8_t**)malloc(nDev * sizeof(int8_t*));
  
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  //float** tempbuff = (float**)malloc(nDev * sizeof(float*));
  
  cudaStream_t* s = (cudaStream_t*)malloc(nDev * sizeof(cudaStream_t));
  
  //int8_t* h_sendbuff = (int8_t*)malloc(size * sizeof(int8_t));
  //int8_t* h_recvbuff = (int8_t*)malloc(size * sizeof(int8_t));
  
  float* h_sendbuff = (float*)malloc(size * sizeof(float));
  float* h_recvbuff = (float*)malloc(size * sizeof(float));
  
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

    float fill_value1 = 2.4;
    thrust::device_ptr<float> dev_ptr1(sendbuff[i]);
    thrust::fill(dev_ptr1, dev_ptr1 + size, fill_value1);

    float fill_value2 = 0;
    thrust::device_ptr<float> dev_ptr2(recvbuff[i]);
    thrust::fill(dev_ptr2, dev_ptr2 + size, fill_value2);



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
  CUDACHECK(cudaMemcpy(h_recvbuff,recvbuff[0],size * sizeof(float),cudaMemcpyDeviceToHost));
  CUDACHECK(cudaDeviceSynchronize());

   //for (int i = 0; i< size; ++i) {
   //  printf("%i\n",h_sendbuff[i]);
   //}
   
   for (int i = 0; i< size; ++i) {
     printf("%f\n",h_sendbuff[i]);
   }

   //for (int i = 0; i< size; ++i) {
   //  printf("%i\n",h_recvbuff[i]);
   //}
 
   for (int i = 0; i< size; ++i) {
     printf("%f\n",h_recvbuff[i]);
   }

   //printf("the first element of the array is: %d \n", h_recvbuff[0]);
   //int count = 0;
   //for(int i=0; i<size; ++i){
   //  if(h_recvbuff[i] != 52){
   //     count++; 
   //     //printf("h_recvbuff[%d] = %d \n", i, h_recvbuff[i]);
   //  }
   //}
   //printf("count = %d \n", count);


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
