#!/usr/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nfs/scistore08/alistgrp/hramezan/nccl/build/lib
export CUDA_HOME=/mnt/nfs/clustersw/shared/cuda/10.1.243 
export CUDA_VISIBLE_DEVICES=4,5,6,7

module load cuda/10.1.243
module load openmpi/4.0.2

#echo $CUDA_VISIBLE_DEVICES
#echo $LD_LIBRARY_PATH
#echo $CUDA_HOME


#make clean
#export DEBUG=1
#make -j src.build CUDA_HOME=/mnt/nfs/clustersw/shared/cuda/10.1.243
