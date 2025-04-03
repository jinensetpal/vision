#!/usr/bin/bash

run=`ls .log/  | wc -l`
echo "Writing to .log/training-$run.log"

PYTHONPATH=~/contrastive-optimization/ \
CUDA_VISIBLE_DEVICES=2,3 \
OMP_NUM_THREADS=200 \
nohup torchrun --nproc_per_node=gpu train.py --model resnet50 --batch-size 512 --lr 0.5 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --epochs 600 --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--train-crop-size 176 --model-ema --val-resize-size 232 --ra-sampler --ra-reps 4 \
--output-dir resnet50_upsampled_causalbnwithgrad_elu --data-path /local/scratch/b/mfdl/datasets/imagenet-1k/images/ > .log/training-$run.log 2>&1 &
