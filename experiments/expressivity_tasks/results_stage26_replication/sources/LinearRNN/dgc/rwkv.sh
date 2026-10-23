#!/bin/bash
#SBATCH -p gpu22
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --job-name=rwkv7
#SBATCH -o rwkv.log
#SBATCH --time=24:00:00
python3 -u train_rwkv7.py \
  --data_dir data/n100 --cuda \
  --rwkv7_depth 2 --rwkv7_head_dim 64 --rwkv7_mode chunk \
  --epochs 999 --max_steps 30000 --batch_size 64 \
  --emb_dim 128 --hidden 256 --dropout 0.1 \
  --lr 3e-5 --weight_decay 1e-4 --grad_clip 0.5 \
  --amp --log_every_steps 500
