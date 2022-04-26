#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh
#python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-01-28/02-49-40/checkpoints/epoch_012.ckpt"

#python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-01-28/02-49-40/checkpoints/epoch_012.ckpt" datamodule.batch_size=2 datamodule.drop_last=True
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=False model.weighted_sum=False model.key='prob' model.sampling='random' model.threshold=0.7
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=False model.key='prob' model.sampling='random' model.threshold=0.7
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=False model.weighted_sum=False model.key='ent' model.sampling='random' model.threshold=0.7
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=False model.key='ent' model.sampling='random' model.threshold=0.7

python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=False model.weighted_sum=False model.key='prob' model.sampling='trust' model.threshold=0.7
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=False model.key='prob' model.sampling='trust' model.threshold=0.7z
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=False model.weighted_sum=False model.key='ent' model.sampling='trust' model.threshold=0.7
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=False model.key='ent' model.sampling='trust' model.threshold=0.7

python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='prob' model.sampling='random' model.threshold=0.5
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='ent' model.sampling='random' model.threshold=0.5
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='prob' model.sampling='trust' model.threshold=0.5
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='ent' model.sampling='trust' model.threshold=0.5

python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='prob' model.sampling='random' model.threshold=0.55
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='ent' model.sampling='random' model.threshold=0.55
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='prob' model.sampling='trust' model.threshold=0.55
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='ent' model.sampling='trust' model.threshold=0.55

python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='prob' model.sampling='random' model.threshold=0.6
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='ent' model.sampling='random' model.threshold=0.6
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='prob' model.sampling='trust' model.threshold=0.6
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='ent' model.sampling='trust' model.threshold=0.6

python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='prob' model.sampling='random' model.threshold=0.65
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='ent' model.sampling='random' model.threshold=0.65
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='prob' model.sampling='trust' model.threshold=0.65
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='ent' model.sampling='trust' model.threshold=0.65

python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='prob' model.sampling='random' model.threshold=0.7
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='ent' model.sampling='random' model.threshold=0.7
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='prob' model.sampling='trust' model.threshold=0.7
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='ent' model.sampling='trust' model.threshold=0.7

python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='prob' model.sampling='random' model.threshold=0.75
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='ent' model.sampling='random' model.threshold=0.75
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='prob' model.sampling='trust' model.threshold=0.75
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='ent' model.sampling='trust' model.threshold=0.75

python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='prob' model.sampling='random' model.threshold=0.8
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='ent' model.sampling='random' model.threshold=0.8
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='prob' model.sampling='trust' model.threshold=0.8
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='ent' model.sampling='trust' model.threshold=0.8

python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='prob' model.sampling='random' model.threshold=0.85
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='ent' model.sampling='random' model.threshold=0.85
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='prob' model.sampling='trust' model.threshold=0.85
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='ent' model.sampling='trust' model.threshold=0.85

python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='ent' model.sampling='random' model.threshold=0.9
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='ent' model.sampling='trust' model.threshold=0.9

python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='ent' model.sampling='random' model.threshold=0.95
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='ent' model.sampling='trust' model.threshold=0.95

python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='ent' model.sampling='random' model.threshold=1.0
python ../test.py ckpt_path="/home/compu/jh/project/colon/logs/runs/2022-02-26/02-35-42/checkpoints/epoch_012.ckpt" model.decide_by_total_probs=True model.weighted_sum=True model.key='ent' model.sampling='trust' model.threshold=1.0