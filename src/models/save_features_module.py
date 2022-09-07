from typing import Any, List
import torch
import torch.nn as nn
import timm
from pytorch_lightning import LightningModule
from src.utils import tensor2np
import wandb
import numpy as np


class SaveFeaturesLitModule(LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        name="vit_base_patch16_224",
        pretrained=True,
        module_type="savefeatures",
        data_ratio=1.0,
        seed=42
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = timm.create_model(
            self.hparams.name, pretrained=self.hparams.pretrained, num_classes=4
        )
        self.discriminator_layer1 = nn.Sequential(
            nn.Linear(self.model.classifier.in_features, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 4),
        ) if 'net' in self.hparams.name else nn.Sequential(
            nn.Linear(self.model.head.in_features, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 4),
        )
        self.discriminator_layer2 = nn.Sequential(
            nn.Linear(self.model.classifier.in_features*2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 3),
        ) if 'net' in self.hparams.name else nn.Sequential(
            nn.Linear(self.model.head.in_features*2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 3),
        )
        # discriminator 구조
        # 레이어 - 드롭아웃 - 레이어
        # 512 512 4 3

    def forward(self, x):  # 4 classification
        return self.discriminator_layer1(
            self.model.forward_head(self.model.forward_features(x.float()), pre_logits=True)
        )

    def get_features(self, x):
        # get features from model
        features = self.model.global_pool(self.model.forward_features(x.float())) if 'densenet' in self.hparams.name else self.model.forward_features(x.float())
        features = features if 'densenet' in self.hparams.name else self.model.forward_head(features, pre_logits=True)
        return features

    def training_step(self, batch,batch_idx):
        pass
    def validation_step(self, batch,batch_idx):
        pass
    def test_step(self, batch,batch_idx):
        x,y=batch
        features=self.get_features(x)
        logits_4cls = self.discriminator_layer1(features)
        preds_4cls = torch.argmax(logits_4cls, dim=1)
        probs_4cls = torch.softmax(logits_4cls, dim=1)
        max_probs = torch.max(probs_4cls, 1)[0]
        
        return {'targets': tensor2np(y), 'preds':tensor2np(preds_4cls),'features': tensor2np(features),'max_probs': tensor2np(max_probs)}

    def test_epoch_end(self, outputs):
        path1='/home/compu/jh/data/voting/features/'
        path2='/home/compu/jh/data/voting/targets/'
        path3='/home/compu/jh/data/voting/preds/'
        path4='/home/compu/jh/data/voting/max_probs/'
        
        # outputs=self.all_gather(outputs)
        
        features = np.vstack([i["features"] for i in outputs])
        targets = np.concatenate([i["targets"] for i in outputs],axis=None)
        preds = np.concatenate([i["preds"] for i in outputs],axis=None)
        max_probs = np.concatenate([i["max_probs"] for i in outputs],axis=None)
        print(self.trainer.datamodule.__class__.__name__.lower()[:-10])
        np.save(f'{path1}{self.trainer.datamodule.__class__.__name__.lower()[:-10]}_{self.hparams.name}_{self.trainer.datamodule.hparams.data_ratio}_seed_{self.hparams.seed}.npy',features)
        np.save(f'{path2}{self.trainer.datamodule.__class__.__name__.lower()[:-10]}_{self.hparams.name}_{self.trainer.datamodule.hparams.data_ratio}_seed_{self.hparams.seed}.npy',targets)
        np.save(f'{path3}{self.trainer.datamodule.__class__.__name__.lower()[:-10]}_{self.hparams.name}_{self.trainer.datamodule.hparams.data_ratio}_seed_{self.hparams.seed}.npy',preds)
        np.save(f'{path4}{self.trainer.datamodule.__class__.__name__.lower()[:-10]}_{self.hparams.name}_{self.trainer.datamodule.hparams.data_ratio}_seed_{self.hparams.seed}.npy',max_probs)