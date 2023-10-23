#!/bin/bash

#SBATCH -G 4
#SBATCH --exclude=ice[100,102-105,107-109,110-134,137-159,160-182,185-189,190-191]
#SBATCH -t 6-02:00
#SBATCH --mem 32G
#SBATCH --chdir=/home/jrick6/repos/simclr/tf2
#SBATCH --job-name=train
#SBATCH --output=/home/jrick6/repos/simclr/tf2/output_train.out

/home/jrick6/.conda/envs/simclr/bin/python  /home/jrick6/repos/simclr/tf2/run.py --train_mode=pretrain \
  --train_batch_size=256 --train_epochs=100 --temperature=0.1 \
  --learning_rate=0.075 --learning_rate_scaling=sqrt --weight_decay=1e-4 \
  --dataset=imagenet2012 --image_size=224 --eval_split=validation \
  --data_dir=/home/jrick6/tensorflow_datasets \
  --model_dir=/home/jrick6/models/simclr_imagenet_run_v3 \
  --use_tpu=False
# 2>&1 | tee /home/jrick6/repos/simclr/tf2/output_run.log