#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

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

  int size = 128*1024*1024;
  //int size = 32*32*32;
  //int size = 8;

  int devs[4] = { 0, 1, 2, 3 };
  //size_t  heapSize = 1024 * 1024 * 1024;
  //cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);


  //allocating and initializing device buffers
  int** sendbuff = (int**)malloc(nDev * sizeof(int*));
  int** recvbuff = (int**)malloc(nDev * sizeof(int*));
  //int** tempbuff1 = (int**)malloc(nDev * sizeof(int*));
  //int** tempbuff2 = (int**)malloc(nDev * sizeof(int*));
  
  //float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  //float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  
  cudaStream_t* s = (cudaStream_t*)malloc(nDev * sizeof(cudaStream_t));
  
  int* h_sendbuff = (int*)malloc(size * sizeof(int));
  int* h_recvbuff = (int*)malloc(size * sizeof(int));
  
  //float* h_sendbuff = (float*)malloc(size * sizeof(float));
  //float* h_recvbuff = (float*)malloc(size * sizeof(float));
  
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc((void**)sendbuff + i, size * sizeof(int)));
    CUDACHECK(cudaMalloc((void**)recvbuff + i, size * sizeof(int)));
    CUDACHECK(cudaMemset(sendbuff[i], 13, size * sizeof(int)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(int)));


    //CUDACHECK(cudaMalloc((void**)tempbuff1 + i, size * sizeof(int)));
    //CUDACHECK(cudaMalloc((void**)tempbuff2 + i, size * sizeof(int)));
    //CUDACHECK(cudaMemset(tempbuff1[i], 0, size * sizeof(int)));
    //CUDACHECK(cudaMemset(tempbuff2[i], 0, size * sizeof(int)));
    //printf("the size of tempbuff1 inside user code is: %d \n", sizeof(tempbuff1[i]));
     
  
    //CUDACHECK(cudaMalloc((void**)sendbuff + i, size * sizeof(float)));
    //CUDACHECK(cudaMalloc((void**)recvbuff + i, size * sizeof(float)));
    //CUDACHECK(cudaMemset(sendbuff[i], 13, size * sizeof(float)));
    //CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
  
  
    CUDACHECK(cudaStreamCreate(s+i));
    CUDACHECK(cudaDeviceSynchronize());
  }

   CUDACHECK(cudaMemcpy(h_sendbuff,sendbuff[0],size * sizeof(int),cudaMemcpyDeviceToHost));
   //cudaMemcpy(h_sendbuff,sendbuff[0],size * sizeof(float),cudaMemcpyDeviceToHost);
   CUDACHECK(cudaDeviceSynchronize());

  //initializing NCCL
  //calling NCCL communication API. Group API is required when using
  //multiple devices per thread
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i){
    //CUDACHECK(cudaSetDevice(i));
    //NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum, comms[i], s[i]));
    //NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclInt8 , ncclSum, comms[i], s[i]));
    NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclInt32 , ncclSum, comms[i], s[i]));
    //NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], (void*)tempbuff1[i], (void*)tempbuff2[i], size, ncclInt8 , ncclSum, comms[i], s[i]));
  }

  NCCLCHECK(ncclGroupEnd());
  CUDACHECK(cudaMemcpy(h_recvbuff,recvbuff[0],size * sizeof(int),cudaMemcpyDeviceToHost));
  //cudaMemcpy(h_recvbuff,recvbuff[0],size * sizeof(float),cudaMemcpyDeviceToHost);
  //CUDACHECK(cudaDeviceSynchronize());

   //for (int i = 0; i< size; ++i) {
   //  printf("%i\n",h_sendbuff[i]);
   //}
   
   //for (int i = 0; i< size; ++i) {
   //  printf("%f\n",h_sendbuff[i]);
   //}

   //for (int i = 0; i< size; ++i) {
   //  printf("%i\n",h_recvbuff[i]);
   //}
 
   //for (int i = 0; i< size; ++i) {
   //  printf("%f\n",h_recvbuff[i]);
   //}

   int count = 0;
   for(int i=0; i<size; ++i){
     if(h_recvbuff[i] != 875836468){
        count++; 
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
    //CUDACHECK(cudaFree(tempbuff1[i]));
    //CUDACHECK(cudaFree(tempbuff2[i]));
  }


  //finalizing NCCL
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);


  printf("Success \n");
  return 0;
}
