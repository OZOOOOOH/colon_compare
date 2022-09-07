from typing import Any, List
import torch
import torch.nn as nn
import timm
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, ConfusionMatrix, F1Score, CohenKappa, Accuracy
from src.utils import get_shuffled_label
from src.utils import get_shuffled_label, get_confmat
import wandb
import numpy as np


class ClassifyCompareLitModule(LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 0.0005,
        t_max: int = 20,
        min_lr: int = 1e-6,
        T_0=15,
        T_mult=2,
        eta_min=1e-6,
        name="vit_base_patch16_224",
        pretrained=True,
        scheduler="ReduceLROnPlateau",
        factor=0.5,
        patience=5,
        eps=1e-08,
        loss_weight=0.5,
        threshold=0.8,
        num_sample=10,
        key="ent",
        sampling="random",
        decide_by_total_probs=False,
        weighted_sum=False,
        module_type="classifycompare",
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

        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.train_acc_compare = Accuracy()
        self.val_acc_compare = Accuracy()
        self.test_acc_compare = Accuracy()
        self.val_acc_best = MaxMetric()
        self.val_acc_compare_best = MaxMetric()
        self.confusion_matrix = ConfusionMatrix(num_classes=4)
        self.f1_score = F1Score(num_classes=4, average="macro")
        self.cohen_kappa = CohenKappa(num_classes=4, weights="quadratic")

    def forward(self, x):  # 4 classification
        return self.discriminator_layer1(self.get_features(x.float()))

    def get_features(self, x):
        # get features from model
        features = self.model.global_pool(self.model.forward_features(x.float())) if 'densenet' in self.hparams.name else self.model.forward_features(x.float())
        features = features if 'densenet' in self.hparams.name else self.model.forward_head(features, pre_logits=True)
        return features

    def get_comparison_list(self, origin, shuffle):
        comparison = []
        for i, j in zip(origin.tolist(), shuffle.tolist()):
            if i > j:
                comparison.append(0)
            elif i == j:
                comparison.append(1)
            else:
                comparison.append(2)
        return torch.tensor(comparison, device=self.device)

    def shuffle_batch(self, x, y):

        indices, shuffle_y = get_shuffled_label(x, y)
        comparison = self.get_comparison_list(y, shuffle_y)

        return indices, comparison, shuffle_y

    def step_classify_compare(self, batch):
        x, y = batch
        features = self.get_features(x)
        logits_4cls = self.discriminator_layer1(features)
        loss_4cls = self.criterion(logits_4cls, y)
        preds_4cls = torch.argmax(logits_4cls, dim=1)

        shuffle_indices, comparison, shuffle_y = self.shuffle_batch(x, y)
        shuffle_features = torch.stack([features[i] for i in shuffle_indices], dim=0)
        # size of shuffle_feature is [16, 768]

        concat_features = torch.cat((features, shuffle_features), dim=1)
        logits_compare = self.discriminator_layer2(concat_features)
        loss_compare = self.criterion(logits_compare, comparison)
        preds_compare = torch.argmax(logits_compare, dim=1)

        loss = loss_4cls + loss_compare * self.hparams.loss_weight

        return loss, preds_4cls, preds_compare, comparison, y

    def training_step(self, batch, batch_idx):

        loss, preds_4cls, preds_compare, comparison, target_4cls = self.step_classify_compare(batch)
        acc = self.train_acc(preds=preds_4cls, target=target_4cls)
        acc_compare = self.train_acc_compare(preds=preds_compare, target=comparison)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc_compare", acc_compare, on_step=True, on_epoch=True, prog_bar=True)
        self.log("LearningRate", self.optimizer.param_groups[0]["lr"])

        return {
            "loss": loss,
            "acc": acc,
            "preds": preds_4cls,
            "targets": target_4cls,
            "acc_compare": acc_compare,
            "preds_compare": preds_compare,
            "comparison": comparison,
        }

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        sch = self.lr_schedulers()
        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val/loss"])

    def validation_step(self, batch, batch_idx):

        loss, preds_4cls, preds_compare, comparison, target_4cls = self.step_classify_compare(batch)
        acc = self.val_acc(preds_4cls, target_4cls)
        acc_compare = self.val_acc_compare(preds=preds_compare, target=comparison)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_compare", acc_compare, on_step=False, on_epoch=True, prog_bar=True)

        return {
            "loss": loss,
            "acc": acc,
            "preds": preds_4cls,
            "targets": target_4cls,
            "acc_compare": preds_compare,
            "preds_compare": preds_compare,
            "comparison": comparison,
        }

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        acc = self.val_acc.compute()
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        acc_compare = self.val_acc_compare.compute()
        self.val_acc_compare_best.update(acc_compare)
        self.log(
            "val/acc_compare_best",
            self.val_acc_compare_best.compute(),
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):

        loss, preds_4cls, preds_compare, comparison, target_4cls = self.step_classify_compare(batch)
        self.confusion_matrix(preds_4cls, target_4cls)
        self.f1_score(preds_4cls, target_4cls)
        self.cohen_kappa(preds_4cls, target_4cls)

        acc = self.test_acc(preds_4cls, target_4cls)
        acc_compare = self.test_acc_compare(preds_compare, comparison)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        self.log("test/acc_compare", acc_compare, on_step=False, on_epoch=True)

        return {
            "loss": loss,
            "acc": acc,
            "preds": preds_4cls,
            "targets": target_4cls,
            "acc_compare": preds_compare,
            "preds_compare": preds_compare,
            "comparison": comparison,
        }

    def test_epoch_end(self, outputs):

        cm = self.confusion_matrix.compute()
        f1 = self.f1_score.compute()
        qwk = self.cohen_kappa.compute()
        p = get_confmat(cm)

        self.logger.experiment.log({"test/conf_matrix": wandb.Image(p)})
        self.log("test/f1_macro", f1, on_step=False, on_epoch=True)
        self.log("test/wqKappa", qwk, on_step=False, on_epoch=True)

        self.test_acc.reset()
        self.confusion_matrix.reset()
        self.f1_score.reset()
        self.cohen_kappa.reset()

    def on_epoch_end(self):
        self.train_acc.reset()
        self.val_acc.reset()
        self.test_acc.reset()
        self.train_acc_compare.reset()
        self.val_acc_compare.reset()
        self.test_acc_compare.reset()

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        self.scheduler = self.get_scheduler()
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": self.scheduler,
                "monitor": "val/loss",
            }

        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def get_scheduler(self):
        schedulers = {
            "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.hparams.factor,
                patience=self.hparams.patience,
                verbose=True,
                eps=self.hparams.eps,
            ),
            "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.hparams.t_max, eta_min=self.hparams.min_lr, last_epoch=-1
            ),
            "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.hparams.T_0,
                T_mult=1,
                eta_min=self.hparams.min_lr,
                last_epoch=-1,
            ),
            "StepLR": torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.1),
            "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95),
        }
        if self.hparams.scheduler not in schedulers:
            raise ValueError(f"Unknown scheduler: {self.hparams.scheduler}")

        return schedulers.get(self.hparams.scheduler, schedulers["ReduceLROnPlateau"])
