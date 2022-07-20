#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh
#python test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-01-28/02-49-40/checkpoints/epoch_012.ckpt"

#python test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-01-28/02-49-40/checkpoints/epoch_012.ckpt" datamodule.batch_size=2 datamodule.drop_last=True

python train.py model.lr= 1e-3