import torch
import torch.nn as nn
import timm
from pytorch_lightning import LightningModule
from torchmetrics import ConfusionMatrix, F1Score, CohenKappa, Accuracy
import numpy as np
import pandas as pd
from utils import tensor2np
from scipy.stats import entropy


class MakeDfLitModule(LightningModule):
    """
    This module is used to make dataframe that has entropy, max_probs, targets, preds about test dataset

    """

    def __init__(
        self,
        lr: float = 1e-4,
        name="vit_base_patch16_224",
        pretrained=True,
        module_type="make_df",
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = timm.create_model(
            self.hparams.name, pretrained=self.hparams.pretrained, num_classes=4
        )
        self.discriminator_layer1 = nn.Sequential(
            nn.Linear(self.model.head.in_features, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 4),
        )
        self.discriminator_layer2 = nn.Sequential(
            nn.Linear(self.model.head.in_features * 2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 3),
        )

        # discriminator 구조
        # 레이어 - 드롭아웃 - 레이어
        # 512 512 4 3

        self.criterion = torch.nn.CrossEntropyLoss()
        self.test_acc = Accuracy()
        self.confusion_matrix = ConfusionMatrix(num_classes=4)
        self.f1_score = F1Score(num_classes=4, average="macro")
        self.cohen_kappa = CohenKappa(num_classes=4, weights="quadratic")
        # self.cnt = SumMetric()

    def forward(self, x):  # 4 classification
        return self.discriminator_layer1(
            self.model.forward_head(
                self.model.forward_features(x.float()), pre_logits=True
            )
        )

    def get_features(self, x):
        """get features from timm models"""
        features = self.model.forward_features(x.float())
        features = self.model.forward_head(features, pre_logits=True)
        return features

    def step(self, batch):
        test_x, test_y = batch
        test_4cls_features = self.get_features(test_x.float())
        test_4cls_logits = self.discriminator_layer1(test_4cls_features)
        test_4cls_loss = self.criterion(test_4cls_logits, test_y)
        test_4cls_preds = torch.argmax(test_4cls_logits, dim=1)
        test_4cls_probs = torch.softmax(test_4cls_logits, dim=1)
        test_4cls_max_probs = torch.max(test_4cls_probs, 1)
        test_4cls_entropy = list(map(lambda x: entropy(x), test_4cls_probs.cpu()))

        losses = test_4cls_loss

        return (
            losses,
            test_4cls_probs,
            test_y,
            test_4cls_preds,
            test_4cls_max_probs,
            test_4cls_entropy,
        )

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):

        loss, probs, targets, preds, max_probs, entropies = self.step(batch)
        return {
            "loss": loss,
            "probs": probs,
            "targets": targets,
            "preds": preds,
            "max_probs": max_probs,
            "entropies": entropies,
        }

    def test_epoch_end(self, outputs):

        probs = [i["probs"] for i in outputs]
        targets = [i["targets"] for i in outputs]
        preds = [i["preds"] for i in outputs]
        max_probs = [i["max_probs"].values for i in outputs]

        all_entropies = np.concatenate(np.array([i["entropies"] for i in outputs]))
        all_max_probs = tensor2np(torch.cat(max_probs))
        all_targets = tensor2np(torch.cat(targets))
        all_probs = tensor2np(torch.cat(probs))
        all_preds = tensor2np(torch.cat(preds))

        df = pd.DataFrame(all_probs, columns=["cls_0", "cls_1", "cls_2", "cls_3"])
        df["ent"] = all_entropies
        df["max_probs"] = all_max_probs
        df["targets"] = all_targets
        df["preds"] = all_preds
        df.to_csv("/home/compu/jh/project/colon_compare/gastric.csv")
