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
  //int size = 32*1024*1024;
 
  int size = 1;
  int devs[4] = { 0, 1, 2, 3 };


  //allocating and initializing device buffers
  int** sendbuff = (int**)malloc(nDev * sizeof(int8_t*));
  int** recvbuff = (int**)malloc(nDev * sizeof(int8_t*));
  cudaStream_t* s = (cudaStream_t*)malloc(nDev * sizeof(cudaStream_t));
  
  int8_t* h_sendbuff = (int8_t*)malloc(size * sizeof(int8_t));
  int8_t* h_recvbuff = (int8_t*)malloc(size * sizeof(int8_t));
  
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc((void**)sendbuff + i, size * sizeof(int8_t)));
    CUDACHECK(cudaMalloc((void**)recvbuff + i, size * sizeof(int8_t)));
    CUDACHECK(cudaMemset(sendbuff[i], 13, size * sizeof(int8_t)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(int8_t)));
    CUDACHECK(cudaStreamCreate(s+i));
    cudaDeviceSynchronize();
  }

   cudaMemcpy(h_sendbuff,sendbuff[0],size * sizeof(int8_t),cudaMemcpyDeviceToHost);
   cudaDeviceSynchronize();
   printf("%i",h_sendbuff[0]);
   printf("\n");


  //initializing NCCL
  //calling NCCL communication API. Group API is required when using
  //multiple devices per thread
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i){
   // NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum, comms[i], s[i]));
    NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclInt8, ncclSum, comms[i], s[i]));
  }

  NCCLCHECK(ncclGroupEnd());



   cudaMemcpy(h_recvbuff,recvbuff[0],size * sizeof(int8_t),cudaMemcpyDeviceToHost);
   cudaDeviceSynchronize();
   printf("%i",h_recvbuff[0]);
   printf("\n");
 

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
  }


  //finalizing NCCL
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);


  printf("Success \n");
  return 0;
}

