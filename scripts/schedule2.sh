#python ../train.py logger.wandb.tags=['tripletGL'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.2 model.margin=-5.0 datamodule.num_workers=0 trainer.devices=\'2,3\'
#python ../train.py logger.wandb.tags=['tripletGL'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.2 model.margin=-2.0 datamodule.num_workers=0 trainer.devices=\'2,3\'
#python ../train.py logger.wandb.tags=['tripletGL'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.2 model.margin=-1.0 datamodule.num_workers=0 trainer.devices=\'2,3\'
#python ../train.py logger.wandb.tags=['tripletGL'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.2 model.margin=-0.5 datamodule.num_workers=0 trainer.devices=\'2,3\'

#python ../train.py logger.wandb.tags=['tripletGL'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.2 model.margin=0.1 datamodule.num_workers=0 trainer.devices=\'2,3\'
#python ../train.py logger.wandb.tags=['tripletGL'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.2 model.margin=0.2 datamodule.num_workers=0 trainer.devices=\'2,3\'
#python ../train.py logger.wandb.tags=['tripletGL'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.2 model.margin=0.3 datamodule.num_workers=0 trainer.devices=\'2,3\'
#python ../train.py logger.wandb.tags=['tripletGL'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.2 model.margin=0.4 datamodule.num_workers=0 trainer.devices=\'2,3\'
#python ../train.py logger.wandb.tags=['tripletGL'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.2 model.margin=0.5 datamodule.num_workers=0 trainer.devices=\'2,3\'
#python ../train.py logger.wandb.tags=['tripletGL'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.2 model.margin=0.6 datamodule.num_workers=0 trainer.devices=\'2,3\'
#python ../train.py logger.wandb.tags=['tripletGL'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.2 model.margin=0.7 datamodule.num_workers=0 trainer.devices=\'2,3\'
#python ../train.py logger.wandb.tags=['tripletGL'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.2 model.margin=0.8 datamodule.num_workers=0 trainer.devices=\'2,3\'
#python ../train.py logger.wandb.tags=['tripletGL'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.2 model.margin=0.9 datamodule.num_workers=0 trainer.devices=\'2,3\'
#python ../train.py logger.wandb.tags=['tripletGL'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.2 model.margin=1.0 datamodule.num_workers=0 trainer.devices=\'2,3\'
# python ../train.py logger.wandb.tags=['triplet'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.2 model.margin=1.5 datamodule.num_workers=0 trainer.devices=\'2,3,4,5,6,7\'
# python ../train.py logger.wandb.tags=['triplet'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.2 model.margin=2.0 datamodule.num_workers=0 trainer.devices=\'2,3,4,5,6,7\'
# python ../train.py logger.wandb.tags=['triplet'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.2 model.margin=1.5 datamodule.num_workers=0 trainer.devices=\'2,3,4,5,6,7\' datamodule.batch_size=32
# python ../train.py logger.wandb.tags=['triplet'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.2 model.margin=2.0 datamodule.num_workers=0 trainer.devices=\'2,3,4,5,6,7\' datamodule.batch_size=32

# python ../train.py logger.wandb.tags=['triplet'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.2 model.margin=5 datamodule.num_workers=0 trainer.devices=\'0,1,2,3,4,5,6,7\'
# python ../train.py logger.wandb.tags=['triplet'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.2 model.margin=7 datamodule.num_workers=0 trainer.devices=\'0,1,2,3,4,5,6,7\'
# python ../train.py logger.wandb.tags=['triplet'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.2 model.margin=5 datamodule.num_workers=0 trainer.devices=\'0,1,2,3,4,5,6,7\' datamodule.batch_size=32
# python ../train.py logger.wandb.tags=['triplet'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.2 model.margin=7 datamodule.num_workers=0 trainer.devices=\'0,1,2,3,4,5,6,7\' datamodule.batch_size=32

# python ../train.py logger.wandb.tags=['tripletGL'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.2 model.margin=3.0 datamodule.num_workers=0 trainer.devices=\'0,1,2,3,4,5,6,7\'
# python ../train.py logger.wandb.tags=['tripletGL'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.2 model.margin=4.0 datamodule.num_workers=0 trainer.devices=\'0,1,2,3,4,5,6,7\'
# python ../train.py logger.wandb.tags=['tripletGL'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.2 model.margin=5.0 datamodule.num_workers=0 trainer.devices=\'0,1,2,3,4,5,6,7\'
# python ../train.py logger.wandb.tags=['arcface'] model.scheduler='CosineAnnealingLR' model.loss_weight=1.0 datamodule.num_workers=0 trainer.devices=\'0,1,2,3,4,5,6,7\' datamodule.batch_size=16 model.scale=32 model.margin=10
# python ../train.py logger.wandb.tags=['arcface'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.5 datamodule.num_workers=0 trainer.devices=\'0,1,2,3,4,5,6,7\' datamodule.batch_size=16 model.scale=32 model.margin=10
# python ../train.py logger.wandb.tags=['arcface'] model.scheduler='CosineAnnealingLR' model.loss_weight=1.0 datamodule.num_workers=0 trainer.devices=\'0,1,2,3,4,5,6,7\' datamodule.batch_size=16 model.scale=16 model.margin=5
# python ../train.py logger.wandb.tags=['arcface'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.5 datamodule.num_workers=0 trainer.devices=\'0,1,2,3,4,5,6,7\' datamodule.batch_size=16 model.scale=16 model.margin=5
# python ../train.py logger.wandb.tags=['only_classify'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.5 model.margin=0.5 datamodule.num_workers=0 trainer.devices=\'0,1,2,3,4,5,6,7\' datamodule.batch_size=32 model.scale=16
# python ../train.py logger.wandb.tags=['only_classify'] model.scheduler='CosineAnnealingLR' model.loss_weight=0.5 model.margin=1.0 datamodule.num_workers=0 trainer.devices=\'0,1,2,3,4,5,6,7\' datamodule.batch_size=16 model.scale=16

python ../train.py -m trainer.strategy='ddp' logger.wandb.group='ddp' logger.wandb.tags=['arcface'] model.scheduler='CosineAnnealingLR' datamodule.num_workers=0 datamodule.batch_size=16 trainer.devices=\'2,3,4,5,6,7\' model.loss_weight=1,0.5,0.2 model.scale=64,32,16,8 model.margin=10,5,2


