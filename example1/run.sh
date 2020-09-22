#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nfs/scistore08/alistgrp/hramezan/nccl/build/lib
export CUDA_HOME=/mnt/nfs/clustersw/shared/cuda/10.1.243
export CUDA_VISIBLE_DEVICES=4,5,6,7

module load cuda/10.1.243
module load openmpi/4.0.2

rm a.out 


#gcc -I/nfs/scistore08/alistgrp/hramezan/nccl/build/include -L/nfs/scistore08/alistgrp/hramezan/nccl/build/lib -lnccl -lcuda -lcudart -g  example1.cu

#g++ -I/nfs/scistore08/alistgrp/hramezan/nccl/build/include -L/nfs/scistore08/alistgrp/hramezan/nccl/build/lib -lnccl -lcuda -lcudart -g  example1.cu

nvcc -I/nfs/scistore08/alistgrp/hramezan/nccl/build/include -L/nfs/scistore08/alistgrp/hramezan/nccl/build/lib -lnccl -lcuda -lcudart -g -G  example1.cu


#gdb ./a.out --x gdb_script
 cuda-gdb ./a.out --x gdb_script


#./a.out

