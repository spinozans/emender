#!/bin/bash
#SBATCH -p gpu22
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -o delta.log

DATA_DIR="${DATA_DIR:-data/n100}"
python3 train_deltanet.py --data_dir "$DATA_DIR" --cuda --epochs 100 --batch_size 128 \
    --emb_dim 128 --hidden_size 256 --layers 1 --num_heads 4 --mode chunk --dropout 0.1 \
    --max_steps 30000 --log_every_steps 500

