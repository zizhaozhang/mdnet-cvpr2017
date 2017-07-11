#!/usr/bin/env bash

export nesterov=true
export randomcrop_type=reflection
export generate_graph=true

export save=logs/ecnet_${depth}_${widen_factor}_${dataset}
mkdir -p $save
CUDA_VISIBLE_DEVICES=$device_id th train.lua | tee $save/log.txt

#cifar 100
#device_id=0 depth=56 widen_factor=12 dataset='cifar100' sh scripts/train_cifar.sh