#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh
#python test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-01-28/02-49-40/checkpoints/epoch_012.ckpt"

#python test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-01-28/02-49-40/checkpoints/epoch_012.ckpt" datamodule.batch_size=2 datamodule.drop_last=True

python train.py model.lr= 1e-3
python train.py model.lr= 1e-4
python train.py model.lr= 1e-5

python train.py model.lr= 1e-3 model.name=swin_large_patch4_window12_384_in22k
python train.py model.lr= 1e-4 model.name=swin_large_patch4_window12_384_in22k
python train.py model.lr= 1e-5 model.name=swin_large_patch4_window12_384_in22k

python train.py model.lr= 1e-3 model.name=swin_large_patch4_window12_384
python train.py model.lr= 1e-4 model.name=swin_large_patch4_window12_384
python train.py model.lr= 1e-5 model.name=swin_large_patch4_window12_384

python train.py model.lr= 1e-3 model.name=swin_base_patch4_window12_384_in22k
python train.py model.lr= 1e-4 model.name=swin_base_patch4_window12_384_in22k
python train.py model.lr= 1e-5 model.name=swin_base_patch4_window12_384_in22k

python train.py model.lr= 1e-3 model.name=swin_base_patch4_window12_384
python train.py model.lr= 1e-4 model.name=swin_base_patch4_window12_384
python train.py model.lr= 1e-5 model.name=swin_base_patch4_window12_384

python train.py model.lr= 1e-3 model.name=beit_base_patch16_384
python train.py model.lr= 1e-4 model.name=beit_base_patch16_384
python train.py model.lr= 1e-5 model.name=beit_base_patch16_384

python train.py model.lr= 1e-3 model.name=cait_m36_384
python train.py model.lr= 1e-4 model.name=cait_m36_384
python train.py model.lr= 1e-5 model.name=cait_m36_384

python train.py model.lr= 1e-3 model.name=cait_m36_384
python train.py model.lr= 1e-4 model.name=cait_m36_384
python train.py model.lr= 1e-5 model.name=cait_m36_384