#!/usr/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nfs/scistore08/alistgrp/hramezan/nccl/build/lib
export CUDA_HOME=/mnt/nfs/clustersw/shared/cuda/10.1.243
export CUDA_VISIBLE_DEVICES=4,5,6,7

module load cuda/10.1.243
module load openmpi/4.0.2

rm a.out

gcc -I../../nccl/build/include/ -L../../nccl/build/lib -lnccl -lcuda -lcudart -lmpi -g  example3.c



gdb ./a.out --x gdb_script
#./a.out


