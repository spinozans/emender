#!/bin/bash
#SBATCH -p gpu22
#SBATCH -c 1
#SBATCH --gres gpu:1
#SBATCH -o rnn.log
DATA_DIR="${DATA_DIR:-data/n100}"
python3 -u train_rnn.py --data_dir "$DATA_DIR" --cuda \
    --epochs 999 --max_steps 30000 --batch_size 256 --emb_dim 128 --hidden 256 --layers 1 \
    --dropout 0.1 --weight_decay 1e-4 --lr 3e-4 --grad_clip 1.0 \
    --hh_shrink 0.3 --post_act_clip 6.0 \
    --log_every_steps 500 \
    --num_workers 8 --prefetch_factor 4 --persistent_workers \
    --amp --tf32
