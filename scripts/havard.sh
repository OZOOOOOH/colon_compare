# python ../train.py model=classify4.yaml datamodule=havard.yaml logger.wandb.tags=['only_classify'] model.scheduler='CosineAnnealingLR' datamodule.num_workers=4 datamodule.batch_size=16 trainer.devices=\'4,5,6,7\' logger.wandb.project='havard'
# python ../train.py model=classifycompare.yaml datamodule=havard.yaml logger.wandb.tags=['classify+compare'] model.scheduler='CosineAnnealingLR' datamodule.num_workers=4 datamodule.batch_size=16 trainer.devices=\'4,5,6,7\' logger.wandb.project='havard'
# python ../test.py model=savefeatures.yaml datamodule=havard.yaml logger=none datamodule.num_workers=8 trainer.devices=[1] ckpt_path=-08-11_13-27-50/checkpoints/epoch_007.ckpt"

