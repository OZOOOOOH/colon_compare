from typing import Any, List
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, ConfusionMatrix, F1Score, CohenKappa, SumMetric, Accuracy
import numpy as np
import pandas as pd
import wandb
from src.datamodules.colon_datamodule import CustomDataset
from src.utils import (
    vote_results,
    get_shuffled_label,
    dist_indexing,
    TripletLoss,
    TripletLossWithGL,
    get_distmat_heatmap,
    get_confmat,
    get_feature_df,
)
from src.utils.loss import *
import copy
from scipy.stats import entropy
import operator
import seaborn as sns
import umap.umap_ as umap
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from pytorch_metric_learning.losses import TripletMarginLoss, ArcFaceLoss
from pytorch_metric_learning.miners import TripletMarginMiner
from pytorch_metric_learning.reducers import ThresholdReducer


class ColonLitModule(LightningModule):
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
        margin=0.5,
        scale=64,
    ):
        super(ColonLitModule, self).__init__()
        self.save_hyperparameters(logger=False)

        self.model = timm.create_model(
            self.hparams.name, pretrained=self.hparams.pretrained, num_classes=4
        )
        # self.compare_layer = nn.Linear(self.model.embed_dim * 2, 3)
        self.discriminator_layer1 = nn.Sequential(
            nn.Linear(self.model.head.in_features, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 4),
        )
        # self.discriminator_layer1 = nn.Sequential(
        #     nn.Linear(self.model.fc.in_features, 512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(256, 4),
        # )
        # self.discriminator_layer2 = nn.Sequential(
        #     nn.Linear(self.model.head.in_features * 2, 512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(256, 3),
        # )
        # self.fc_layer = nn.Sequential(
        #     nn.Linear(self.model.head.in_features, 512),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )
        # discriminator 구조
        # 레이어 - 드롭아웃 - 레이어
        # 512 512 4 3

        self.distance = LpDistance()
        # self.distance = CosineSimilarity()
        self.reducer = ThresholdReducer(low=0)
        self.triplet_loss = TripletMarginLoss(
            margin=self.hparams.margin, distance=self.distance, reducer=self.reducer
        )
        self.miner = TripletMarginMiner(margin=0.2, distance=self.distance, type_of_triplets="hard")

        # self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.889, 0.734, 0.573, 0.802]).cuda()) # weighted CE
        self.criterion = nn.CrossEntropyLoss()
        self.arcface = ArcFaceLoss(
            margin=self.hparams.margin,
            num_classes=4,
            embedding_size=self.model.head.in_features,
            scale=self.hparams.scale,
        )
        # self.arcface = ArcMarginProduct(
        #     in_features=self.model.head.in_features,
        #     out_features=4,
        #     s=opt.scale,
        #     m=opt.margin,
        #     easy_margin=opt.easy_margin,
        #     ls_eps=opt.ls_eps,
        # )

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        # self.train_acc_compare = Accuracy()
        # self.val_acc_compare = Accuracy()
        # self.test_acc_compare = Accuracy()
        self.val_acc_best = MaxMetric()
        # self.val_acc_compare_best = MaxMetric()
        self.confusion_matrix = ConfusionMatrix(num_classes=4)
        self.f1_score = F1Score(num_classes=4, average="macro")
        self.cohen_kappa = CohenKappa(num_classes=4, weights="quadratic")
        self.cnt = SumMetric()

    def forward(self, x):  # 4 classification
        return self.model.forward(x.float())

    def get_embedding_data(self, df, targets):
        targets = targets.detach().cpu().numpy()
        embedding = umap.UMAP().fit_transform(df.drop("LABEL", 1), y=targets)
        fig, ax = plt.subplots(1, figsize=(14, 10))
        plt.scatter(*embedding.T, s=5, c=targets, cmap="Spectral", alpha=1.0),
        plt.setp(ax, xticks=[], yticks=[])
        cbar = plt.colorbar(boundaries=np.arange(11) - 0.5)
        cbar.set_ticks(np.arange(4))
        cbar.set_ticklabels(["BN_0", "WD_1", "MD_2", "PD_3"])
        plt.title(f"epoch {self.current_epoch}")
        return fig

    def step(self, batch):
        x, y = batch
        features = self.model.forward_features(x.float())
        logits = self.discriminator_layer1(features)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        indices, comparison, shuffle_y = self.shuffle_batch(x, y)
        shuffle_features = [features[i] for i in indices]
        shuffle_features = torch.stack(shuffle_features, dim=0)
        # size of shuffle_feature is [16, 768]

        concat_features_with_shuffled = torch.cat((features, shuffle_features), dim=1)

        logits_compare = self.discriminator_layer2(concat_features_with_shuffled)
        loss_compare = self.criterion(logits_compare, comparison)
        preds_compare = torch.argmax(logits_compare, dim=1)

        losses = loss + loss_compare * self.hparams.loss_weight

        return losses, preds, y, preds_compare, comparison

    def step_after_concat(self, batch):  # after concat
        x, y = batch
        features = self.model.forward_features(x.float())
        logits = self.discriminator_layer1(features)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        indices, comparison, shuffle_y = self.shuffle_batch(x, y)
        shuffle_features = [features[i] for i in indices]
        shuffle_features = torch.stack(shuffle_features, dim=0)
        # size of shuffle_feature is [16, 768]
        features = self.fc_layer(features)
        shuffle_features = self.fc_layer(shuffle_features)

        concat_features_with_shuffled = torch.cat((features, shuffle_features), dim=1)

        logits_compare = self.discriminator_layer3(concat_features_with_shuffled)
        loss_compare = self.criterion(logits_compare, comparison)
        preds_compare = torch.argmax(logits_compare, dim=1)

        losses = loss + loss_compare * self.hparams.loss_weight

        return losses, preds, y, preds_compare, comparison

    # def step_test0(self, batch, stage):  # only classification
    #     x, y = batch
    #     logits = self.forward(x)
    #     loss = self.criterion(logits, y)
    #     preds = torch.argmax(logits, dim=1)
    #     return loss, logits, preds, y

    def step_test0(self, batch):  # only classification
        x, y = batch
        features = self.model.forward_features(x.float())
        logits_4cls = self.discriminator_layer1(features)
        loss_4cls = self.criterion(logits_4cls, y)
        preds_4cls = torch.argmax(logits_4cls, dim=1)
        loss = loss_4cls

        return loss, logits_4cls, preds_4cls, y, features

    def step_test1(self, batch):  # classification + compare
        x, y = batch
        features = self.model.forward_features(x.float())
        logits_4cls = self.discriminator_layer1(features)
        loss_4cls = self.criterion(logits_4cls, y)
        preds_4cls = torch.argmax(logits_4cls, dim=1)

        shuffle_indices, comparison, shuffle_y = self.shuffle_batch(x, y)
        shuffle_features = torch.stack([features[i] for i in shuffle_indices], dim=0)
        # size of shuffle_feature is [16, 768]

        concat_features = torch.cat((features, shuffle_features), dim=1)
        logits_compare = self.discriminator_layer2(concat_features)
        # logits_compare = self.compare_layer(concat_features)
        loss_compare = self.criterion(logits_compare, comparison)
        preds_compare = torch.argmax(logits_compare, dim=1)

        loss = loss_4cls + loss_compare * self.hparams.loss_weight

        return (
            loss,
            logits_4cls,
            logits_compare,
            preds_4cls,
            preds_compare,
            comparison,
            y,
            shuffle_y,
        )

    def step_test2(
        self, batch
    ):  # random sampling or convinced sampling or weighted_sum or thresholding or
        x, y = batch
        features = self.model.forward_features(x.float())
        logits_4cls = self.discriminator_layer1(features)
        loss_4cls = self.criterion(logits_4cls, y)
        preds_4cls = torch.argmax(logits_4cls, dim=1)
        probs_4cls = torch.softmax(logits_4cls, dim=1)
        max_probs_4cls = torch.max(probs_4cls, 1)
        origin_preds_4cls = copy.deepcopy(preds_4cls)
        entropy_4cls = list(map(lambda i: entropy(i), probs_4cls.detach().cpu().numpy()))
        imgs_0, imgs_1, imgs_2, imgs_3 = (
            self.bring_random_trained_data(x)
            if self.hparams.sampling == "random"
            else self.bring_convinced_trained_data(x)
        )

        total_imgs = [imgs_0, imgs_1, imgs_2, imgs_3]
        cnt_correct_diff, new_preds_4cls = self.predict_using_voting_cosine(
            entropy_4cls, max_probs_4cls, features, total_imgs, probs_4cls, preds_4cls, y
        )
        cnt_diff = sum(x != y for x, y in zip(origin_preds_4cls, new_preds_4cls))
        # print(f'length of preds : {len(origin_preds_4cls)} // The number of changed : {cnt_diff}')

        # losses = loss + loss_compare * self.hparams.loss_weight
        loss = loss_4cls

        return loss, logits_4cls, origin_preds_4cls, new_preds_4cls, y, cnt_diff, cnt_correct_diff

    def step_test3(self, batch):  # only classification
        x, y = batch
        features = self.model.forward_features(x.float())
        shuffle_indices, comparison, shuffle_y = self.shuffle_batch(x, y)
        shuffle_features = torch.stack([features[i] for i in shuffle_indices], dim=0)
        concat_features = torch.cat((features, shuffle_features), dim=1)
        logits_compare = self.discriminator_layer2(concat_features)
        loss_compare = self.criterion(logits_compare, comparison)
        preds_compare = torch.argmax(logits_compare, dim=1)
        return loss_compare, logits_compare, preds_compare, comparison

    def step_triplet(self, batch):
        x, y = batch
        features = self.model.forward_features(x.float())
        logits_4cls = self.discriminator_layer1(features)
        loss_4cls = self.criterion(logits_4cls, y)
        preds_4cls = torch.argmax(logits_4cls, dim=1)

        shuffle_indices, comparison, shuffle_y = self.shuffle_batch(x, y)
        shuffle_features = torch.stack([features[i] for i in shuffle_indices], dim=0)
        dist_matrix = torch.cdist(features, shuffle_features, p=2)
        y_idx_groupby = [torch.where(y == i)[0].tolist() for i in range(4)]
        dist_indices = dist_indexing(y, shuffle_y, y_idx_groupby, dist_matrix)
        dist_features = torch.stack([features[i] for i in dist_indices], dim=0)
        concat_features = torch.cat((features, dist_features), dim=1)

        logits_compare = self.discriminator_layer2(concat_features)
        loss_compare = self.criterion(logits_compare, comparison)
        preds_compare = torch.argmax(logits_compare, dim=1)

        loss = loss_4cls + loss_compare * self.hparams.loss_weight

        return loss, preds_4cls, preds_compare, comparison, y

    def step_triplet2(self, batch):
        x, y = batch
        features = self.model.forward_head(self.model.forward_features(x.float()), pre_logits=True)
        # triplet = TripletLossWithGL(margin=self.hparams.margin)
        # loss_triplet, dist_matrix, cnt = triplet(features, y)

        logits_4cls = self.discriminator_layer1(features)
        loss_4cls = self.criterion(logits_4cls, y)
        preds_4cls = torch.argmax(logits_4cls, dim=1)

        # loss = loss_4cls + loss_triplet * self.hparams.loss_weight
        return loss_4cls, preds_4cls, y
        # return loss, loss_4cls, loss_triplet, preds_4cls, y, dist_matrix, features, cnt

    def step_triplet_pml(self, batch):
        x, y = batch
        features = self.model.forward_features(x.float())
        features = self.model.forward_head(features, pre_logits=True)

        # features=features.squeeze()
        indices = self.miner(features, y)
        # a,p,n
        loss_triplet = self.triplet_loss(features, y, indices)
        logits_4cls = self.discriminator_layer1(features)
        loss_4cls = self.criterion(logits_4cls, y)
        preds_4cls = torch.argmax(logits_4cls, dim=1)

        loss = loss_4cls + loss_triplet * self.hparams.loss_weight

        return loss, loss_4cls, loss_triplet, preds_4cls, y, features
        # return loss_4cls, preds_4cls, y, features

    def step_arcface(self, batch):
        x, y = batch
        features = self.model.forward_features(x.float())
        features = self.model.forward_head(features, pre_logits=True)

        loss_arcface = self.arcface(features, y)

        logits_4cls = self.discriminator_layer1(features)
        loss_4cls = self.criterion(logits_4cls, y)
        preds_4cls = torch.argmax(logits_4cls, dim=1)

        loss = loss_4cls + loss_arcface * self.hparams.loss_weight

        return loss, loss_4cls, loss_arcface, preds_4cls, y, features

    def step_only_classify(self, batch):
        x, y = batch
        features = self.model.forward_features(x.float())
        logits_4cls = self.model.forward_head(features, pre_logits=False)

        # logits_4cls = self.discriminator_layer1(features)
        loss_4cls = self.criterion(logits_4cls, y)
        preds_4cls = torch.argmax(logits_4cls, dim=1)

        loss = loss_4cls

        return loss, preds_4cls, y, features


    def training_step(self, batch, batch_idx):
        # loss_4cls, preds_4cls, target_4cls = self.step_triplet2(batch)
        loss, loss_4cls, loss_arcface, preds_4cls, target_4cls, features = self.step_arcface(batch)
        acc = self.train_acc(preds=preds_4cls, target=target_4cls)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/loss_4cls", loss_4cls, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_arcface", loss_arcface, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("LearningRate", self.optimizer.param_groups[0]["lr"])

        # self.cnt(cnt)

        # dist_heatmap = get_distmat_heatmap(dist_matrix, target_4cls)
        # self.logger.experiment.log({"train/dist_matrix": wandb.Image(dist_heatmap)})

        return {
            "loss": loss,
            "acc": acc,
            "preds": preds_4cls,
            "targets": target_4cls,
            # "features": features,
            # "cnt": cnt,
        }

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        # targets = torch.cat([i["targets"] for i in outputs], 0)
        # features = torch.cat([i["features"] for i in outputs], 0)
        # df = get_feature_df(features, targets)
        # single_values_in_batch = self.cnt.compute()
        # self.log("train/cnt", single_values_in_batch)
        # self.cnt.reset()

        # self.logger.experiment.log(
        #     {
        #         f"train/features/epoch_{self.current_epoch}": wandb.Table(
        #             columns=df.columns.to_list(), data=df.values
        #         )
        #     }
        # )

        # self.logger.experiment.log({"train/umap": wandb.Image(self.get_embedding_data(df,targets))})
        sch = self.lr_schedulers()
        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val/loss"])

    def validation_step(self, batch, batch_idx):
        # loss_4cls, preds_4cls, target_4cls = self.step_triplet2(batch)
        loss, loss_4cls, loss_arcface, preds_4cls, target_4cls, features = self.step_arcface(batch)
        # loss, preds_4cls, target_4cls, features = self.step_only_classify(batch)

        acc = self.val_acc(preds_4cls, target_4cls)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss_4cls", loss_4cls, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val/loss_arcface", loss_arcface, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # dist_heatmap = get_distmat_heatmap(dist_matrix, target_4cls)
        # self.logger.experiment.log({"val/dist_matrix": wandb.Image(dist_heatmap)})

        return {
            "loss": loss,
            "acc": acc,
            "preds": preds_4cls,
            "targets": target_4cls,
            # "features": features,
        }

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]

        # targets = torch.cat([i["targets"] for i in outputs], 0)
        # features = torch.cat([i["features"] for i in outputs], 0)
        # df = get_feature_df(features, targets)
        # self.logger.experiment.log({"val/umap": wandb.Image(self.get_embedding_data(df,targets))})

        acc = self.val_acc.compute()
        self.val_acc.reset()
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # loss_4cls, preds_4cls, target_4cls = self.step_triplet2(batch)
        loss, loss_4cls, loss_arcface, preds_4cls, target_4cls, features = self.step_arcface(batch)
        # loss, preds_4cls, target_4cls, features = self.step_only_classify(batch)

        acc = self.test_acc(preds_4cls, target_4cls)
        # dist_heatmap = get_distmat_heatmap(dist_matrix, target_4cls)
        # self.logger.experiment.log({"test/dist_matrix": wandb.Image(dist_heatmap)})
        self.confusion_matrix(preds_4cls, target_4cls)
        self.f1_score(preds_4cls, target_4cls)
        self.cohen_kappa(preds_4cls, target_4cls)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/loss_4cls", loss_4cls, on_step=False, on_epoch=True)
        self.log("test/loss_arcface", loss_arcface, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        return {
            "loss": loss,
            "acc": acc,
            "preds": preds_4cls,
            "targets": target_4cls,
            # "features": features,
        }

    def test_epoch_end(self, outputs):
        # pass
        # preds = torch.cat([i["preds"] for i in outputs], 0)
        # targets = torch.cat([i["targets"] for i in outputs], 0)
        # features = torch.cat([i["features"] for i in outputs], 0)
        # df = get_feature_df(features, targets)
        cm = self.confusion_matrix.compute()
        f1 = self.f1_score.compute()
        qwk = self.cohen_kappa.compute()

        self.log("test/f1_macro", f1, on_step=False, on_epoch=True)
        self.log("test/wqKappa", qwk, on_step=False, on_epoch=True)

        p = get_confmat(cm)
        self.logger.experiment.log({"test/conf_matrix": wandb.Image(p)})
        # self.logger.experiment.log(
        #     {"test/features": wandb.Table(columns=df.columns.to_list(), data=df.values)}
        # )
        # self.logger.experiment.log({"val/umap": wandb.Image(self.get_embedding_data(df,targets))})

        self.confusion_matrix.reset()
        self.f1_score.reset()
        self.cohen_kappa.reset()

    def on_epoch_end(self):
        self.train_acc.reset()
        self.val_acc.reset()
        self.test_acc.reset()
        # self.train_acc_compare.reset()
        # self.val_acc_compare.reset()
        # self.test_acc_compare.reset()

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
            "None": torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1**epoch
            ),
        }
        if self.hparams.scheduler not in schedulers:
            raise ValueError(f"Unknown scheduler: {self.hparams.scheduler}")

        return schedulers.get(self.hparams.scheduler, schedulers["ReduceLROnPlateau"])
