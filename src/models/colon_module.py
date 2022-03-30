from typing import Any, List
import torch
import torch.nn as nn
import timm
from pytorch_lightning import plugins
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from src.datamodules.colon_datamodule import CustomDataset
import copy
from scipy.stats import entropy


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
            name='vit_base_patch16_224',
            pretrained=True,
            scheduler='ReduceLROnPlateau',
            factor=0.5,
            patience=5,
            eps=1e-08,
            loss_weight=0.5,
            threshold=0.8,
            num_sample=10,
            key='ent'

    ):
        super(ColonLitModule, self).__init__()
        self.save_hyperparameters(logger=False)

        self.model = timm.create_model(self.hparams.name, pretrained=self.hparams.pretrained, num_classes=4)
        self.compare_layer = nn.Linear(self.model.embed_dim * 2, 3)
        self.discriminator_layer1 = nn.Sequential(
            nn.Linear(self.model.embed_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 4),
        )
        self.discriminator_layer2 = nn.Sequential(
            nn.Linear(self.model.embed_dim * 2, 512),
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

    def forward(self, x):
        return self.discriminator_layer1(self.model.forward_features(x.float()))

    def shuffle(self, x, y):

        z = [list(z) for z in zip(x, y)]
        z = list(enumerate(z))
        z = random.sample(z, len(z))
        indices, z = zip(*z)
        indices = list(indices)
        z = list(z)

        tmp1 = [i[0] for i in z]
        tmp2 = [i[1] for i in z]

        shuffle_x = torch.stack(tmp1, dim=0)
        shuffle_y = torch.stack(tmp2, dim=0)

        # origin > shuffle : 0
        # origin = shuffle : 1
        # origin < shuffle : 2
        comparison = []
        for i, j in zip(y.tolist(), shuffle_y.tolist()):

            if i > j:
                comparison.append(0)
            elif i == j:
                comparison.append(1)
            else:
                comparison.append(2)
        comparison = torch.tensor(comparison, device=self.device)
        return indices, comparison, shuffle_y

    def bring_trained_data(self):
        self.trainer.datamodule.setup()
        train_img_path = self.trainer.datamodule.train_dataloader().dataset.image_id
        train_img_labels = self.trainer.datamodule.train_dataloader().dataset.labels
        random_idx_label0 = np.random.choice(np.where(train_img_labels == 0)[0], self.hparams.num_sample, replace=False)
        random_idx_label1 = np.random.choice(np.where(train_img_labels == 1)[0], self.hparams.num_sample, replace=False)
        random_idx_label2 = np.random.choice(np.where(train_img_labels == 2)[0], self.hparams.num_sample, replace=False)
        random_idx_label3 = np.random.choice(np.where(train_img_labels == 3)[0], self.hparams.num_sample, replace=False)

        random_10_train_path0 = train_img_path[random_idx_label0]
        random_10_train_path1 = train_img_path[random_idx_label1]
        random_10_train_path2 = train_img_path[random_idx_label2]
        random_10_train_path3 = train_img_path[random_idx_label3]

        random_10_train_label0 = train_img_labels[random_idx_label0]
        random_10_train_label1 = train_img_labels[random_idx_label1]
        random_10_train_label2 = train_img_labels[random_idx_label2]
        random_10_train_label3 = train_img_labels[random_idx_label3]

        df_0 = pd.DataFrame({'path': random_10_train_path0,
                             'class': random_10_train_label0
                             })
        df_1 = pd.DataFrame({'path': random_10_train_path1,
                             'class': random_10_train_label1
                             })
        df_2 = pd.DataFrame({'path': random_10_train_path2,
                             'class': random_10_train_label2
                             })
        df_3 = pd.DataFrame({'path': random_10_train_path3,
                             'class': random_10_train_label3
                             })
        dataloader_0 = DataLoader(
            CustomDataset(df_0, self.trainer.datamodule.train_transform),
            batch_size=10,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            shuffle=False,
        )
        dataloader_1 = DataLoader(
            CustomDataset(df_1, self.trainer.datamodule.train_transform),
            batch_size=10,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            shuffle=False,
        )
        dataloader_2 = DataLoader(
            CustomDataset(df_2, self.trainer.datamodule.train_transform),
            batch_size=10,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            shuffle=False,
        )
        dataloader_3 = DataLoader(
            CustomDataset(df_3, self.trainer.datamodule.train_transform),
            batch_size=10,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            shuffle=False,
        )

        return next(iter(dataloader_0)), next(iter(dataloader_1)), next(iter(dataloader_2)), next(
            iter(dataloader_3))

    def step(self, batch):
        x, y = batch
        # logits = self.forward(x)
        features = self.model.forward_features(x.float())
        # logits = self.model.head(features)
        logits = self.discriminator_layer1(features)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        indices, comparison, shuffle_y = self.shuffle(x, y)
        shuffle_features = [features[i] for i in indices]
        shuffle_features = torch.stack(shuffle_features, dim=0)
        # size of shuffle_feature is [16, 768]

        concat_features = torch.cat((features, shuffle_features), dim=1)

        logits_compare = self.discriminator_layer2(concat_features)
        # logits_compare = self.compare_layer(concat_features)
        loss_compare = self.criterion(logits_compare, comparison)
        preds_compare = torch.argmax(logits_compare, dim=1)

        losses = loss + loss_compare * self.hparams.loss_weight

        return losses, preds, y, preds_compare, comparison

    def compare_func(self, idx, features, imgs, labels):
        # compare trained and test labels
        result_list = []
        for img, label in zip(imgs, labels):
            trained_features = self.model.forward_features(img.unsqueeze(0).float())
            concat_test_with_trained_features = torch.cat((features[idx].unsqueeze(0), trained_features), dim=1)
            logits_compare = self.discriminator_layer2(concat_test_with_trained_features)
            pred = torch.argmax(logits_compare, dim=1)
            result_list.append(pred)
        return result_list

    def step_test1(self, batch):
        x, y = batch
        features = self.model.forward_features(x.float())
        logits = self.discriminator_layer1(features)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        indices, comparison, shuffle_y = self.shuffle(x, y)
        shuffle_features = [features[i] for i in indices]
        shuffle_features = torch.stack(shuffle_features, dim=0)
        # size of shuffle_feature is [16, 768]

        concat_features = torch.cat((features, shuffle_features), dim=1)

        logits_compare = self.discriminator_layer2(concat_features)
        # logits_compare = self.compare_layer(concat_features)
        loss_compare = self.criterion(logits_compare, comparison)
        preds_compare = torch.argmax(logits_compare, dim=1)

        losses = loss + loss_compare * self.hparams.loss_weight

        return losses, logits, logits_compare, preds, preds_compare, comparison, y, shuffle_y

    def step_test2(self, batch):
        test_x, test_y = batch
        test_4cls_features = self.model.forward_features(test_x.float())
        test_4cls_logits = self.discriminator_layer1(test_4cls_features)
        test_4cls_loss = self.criterion(test_4cls_logits, test_y)
        test_4cls_preds = torch.argmax(test_4cls_logits, dim=1)
        test_4cls_probs = torch.softmax(test_4cls_logits, dim=1)
        test_4cls_max_probs = torch.max(test_4cls_probs, 1)
        test_4cls_origin_preds = copy.deepcopy(test_4cls_preds)
        test_4cls_entropy = list(map(lambda x: entropy(x), test_4cls_probs.cpu()))
        batch0, batch1, batch2, batch3 = self.bring_trained_data()
        imgs_0, labels_0 = batch0
        imgs_1, labels_1 = batch1
        imgs_2, labels_2 = batch2
        imgs_3, labels_3 = batch3
        labels_0 = labels_0.type_as(test_y)
        labels_1 = labels_1.type_as(test_y)
        labels_2 = labels_2.type_as(test_y)
        labels_3 = labels_3.type_as(test_y)
        imgs_0 = imgs_0.type_as(test_x)
        imgs_1 = imgs_1.type_as(test_x)
        imgs_2 = imgs_2.type_as(test_x)
        imgs_3 = imgs_3.type_as(test_x)
        # bring data from train loader and convert to tensor
        cnt_correct_diff = 0
        # count (original preds != voting preds)

        key = test_4cls_entropy if self.hparams.key == 'ent' else test_4cls_max_probs
        for idx, value in enumerate(key):
            if value < self.hparams.threshold:
                # self.hparams.prob is a threshold (prob)
                vote_cnt_0 = 0
                vote_cnt_1 = 0
                vote_cnt_2 = 0
                vote_cnt_3 = 0
                vote_cnt_else = 0

                result_0 = self.compare_func(idx, test_4cls_features, imgs_0, labels_0)
                result_1 = self.compare_func(idx, test_4cls_features, imgs_1, labels_1)
                result_2 = self.compare_func(idx, test_4cls_features, imgs_2, labels_2)
                result_3 = self.compare_func(idx, test_4cls_features, imgs_3, labels_3)
                # compare train imgs with test imgs // batch size
                result_0 = torch.cat(result_0, dim=0)
                result_1 = torch.cat(result_1, dim=0)
                result_2 = torch.cat(result_2, dim=0)
                result_3 = torch.cat(result_3, dim=0)

                for i in result_0:
                    if i == 0:
                        vote_cnt_1 += 1
                        vote_cnt_2 += 1
                        vote_cnt_3 += 1
                    elif i == 1:
                        vote_cnt_0 += 1
                    else:
                        vote_cnt_else += 1
                for i in result_1:
                    if i == 0:
                        vote_cnt_2 += 1
                        vote_cnt_3 += 1
                    elif i == 1:
                        vote_cnt_1 += 1
                    elif i == 2:
                        vote_cnt_0 += 1

                for i in result_2:
                    if i == 0:
                        vote_cnt_3 += 1
                    elif i == 1:
                        vote_cnt_2 += 1
                    elif i == 2:
                        vote_cnt_0 += 1
                        vote_cnt_1 += 1
                for i in result_3:
                    if i == 0:
                        vote_cnt_else += 1
                    elif i == 1:
                        vote_cnt_3 += 1
                    elif i == 2:
                        vote_cnt_0 += 1
                        vote_cnt_1 += 1
                        vote_cnt_2 += 1

                vote_cls = np.argmax([vote_cnt_0, vote_cnt_1, vote_cnt_2, vote_cnt_3])
                # class based on voting
                if test_4cls_preds[idx].cpu().detach().item() != vote_cls:
                    if vote_cls == test_y[idx]:
                        cnt_correct_diff += 1
                    print()
                    print(f'True label: {test_y[idx]}')
                    print(f'Predict label: {test_4cls_preds[idx]}')
                    print(f'vote_cnt_0:{vote_cnt_0}')
                    print(f'vote_cnt_1:{vote_cnt_1}')
                    print(f'vote_cnt_2:{vote_cnt_2}')
                    print(f'vote_cnt_3:{vote_cnt_3}')
                    print(f'vote_cnt_else:{vote_cnt_else}')
                    print(f'vote_cls:{vote_cls}')
                    print()
                test_4cls_preds[idx] = torch.Tensor([vote_cls]).type_as(test_y)

        new_preds = test_4cls_preds

        cnt_diff = sum(x != y for x, y in zip(test_4cls_origin_preds, new_preds))
        # print(f'length of preds : {len(test_4cls_origin_preds)} // The number of changed : {cnt_diff}')

        # losses = loss + loss_compare * self.hparams.loss_weight
        losses = test_4cls_loss

        return losses, test_4cls_logits, test_4cls_origin_preds, new_preds, test_y, cnt_diff, cnt_correct_diff
        # return losses, logits, logits_compare, preds, preds_compare, comparison, y, shuffle_y

    def step_test3(self, batch):
        test_x, test_y = batch
        test_4cls_features = self.model.forward_features(test_x.float())
        test_4cls_logits = self.discriminator_layer1(test_4cls_features)
        test_4cls_loss = self.criterion(test_4cls_logits, test_y)
        test_4cls_preds = torch.argmax(test_4cls_logits, dim=1)
        test_4cls_probs = torch.softmax(test_4cls_logits, dim=1)
        test_4cls_max_probs = torch.max(test_4cls_probs, 1)
        test_4cls_entropy = list(map(lambda x: entropy(x), test_4cls_probs.cpu()))

        losses = test_4cls_loss

        return losses, test_4cls_probs, test_y, test_4cls_preds, test_4cls_max_probs, test_4cls_entropy

    def training_step(self, batch, batch_idx):
        loss, preds, targets, preds_compare, comparison = self.step(batch)
        acc = self.train_acc(preds=preds, target=targets)
        acc_compare = self.train_acc_compare(preds=preds_compare, target=comparison)
        # sch = self.lr_schedulers()
        # if isinstance(sch, torch.optim.lr_scheduler.CosineAnnealingLR):
        #     sch.step()
        # if isinstance(sch, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
        #     sch.step()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc_compare", acc_compare, on_step=True, on_epoch=True, prog_bar=True)
        self.log("LearningRate", self.optimizer.param_groups[0]['lr'])

        logs = {
            "loss": loss,
            "acc": acc,
            "preds": preds,
            "targets": targets,
            "acc_compare": preds_compare,
            "preds_compare": preds_compare,
            "comparison": comparison
        }

        return logs

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val/loss"])

    def validation_step(self, batch, batch_idx):

        loss, preds, targets, preds_compare, comparison = self.step(batch)

        acc = self.val_acc(preds, targets)
        acc_compare = self.val_acc_compare(preds=preds_compare, target=comparison)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_compare", acc_compare, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "acc": acc, "preds": preds, "targets": targets, "acc_compare": preds_compare,
                "preds_compare": preds_compare, "comparison": comparison}

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        acc = self.val_acc.compute()
        self.val_acc_best.update(acc)

        acc_compare = self.val_acc_compare.compute()
        self.val_acc_compare_best.update(acc_compare)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        self.log("val/acc_compare_best", self.val_acc_compare_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # loss, logits, logits_compare, preds, preds_compare, comparison, targets, targets_shuffle = self.step_test2(batch)
        loss, logits, origin_preds, new_preds, targets, cnt_diff, cnt_correct_diff = self.step_test2(batch)
        origin_acc = self.test_acc(origin_preds, targets)
        new_acc = self.test_acc(new_preds, targets)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/origin_acc", origin_acc, on_step=False, on_epoch=True)
        self.log("test/new_acc", new_acc, on_step=False, on_epoch=True)

        return {"loss": loss, "origin_acc": origin_acc, "new_acc": new_acc, "origin_preds": origin_preds,
                "new_preds": new_preds, "targets": targets, "cnt_diff": cnt_diff, "cnt_correct_diff": cnt_correct_diff}

    # def test_step(self, batch, batch_idx):
    #     loss, probs, targets, preds, max_probs, entropies = self.step_test3(batch)
    #
    #     acc = self.test_acc(preds, targets)
    #
    #     # self.log("test/probs", probs, on_step=True, on_epoch=False)
    #     # self.log("test/targets", targets, on_step=True, on_epoch=False)
    #     # self.log("test/preds", preds, on_step=True, on_epoch=False)
    #     # self.log("test/max_probs", max_probs.values, on_step=True, on_epoch=False)
    #     # self.log("test/entropy", torch.Tensor(entropy), on_step=True, on_epoch=False)
    #
    #     return {"loss": loss, "acc": acc, "probs": probs, "targets": targets, "preds": preds, "max_probs": max_probs,
    #             "entropies": entropies}

    def test_epoch_end(self, outputs):
        # pass
        cnt_diff = sum([i['cnt_diff'] for i in outputs])
        cnt_correct_diff = sum([i['cnt_correct_diff'] for i in outputs])
        cnt_diff = cnt_diff.sum()
        self.log("test/cnt_diff", cnt_diff, on_epoch=True, on_step=False, reduce_fx='sum')
        self.log("test/cnt_correct_diff", cnt_correct_diff, on_epoch=True, on_step=False, reduce_fx='sum')

    # def test_epoch_end(self, outputs):
    #     # pass
    #     probs = [i['probs'] for i in outputs]
    #     targets = [i['targets'] for i in outputs]
    #     preds = [i['preds'] for i in outputs]
    #     max_probs = [i['max_probs'].values for i in outputs]
    #
    #     all_entropies = np.concatenate(np.array([i['entropy'] for i in outputs]))
    #     all_max_probs = torch.cat(max_probs).detach().cpu().numpy()
    #     all_targets = torch.cat(targets).detach().cpu().numpy()
    #     all_probs = torch.cat(probs).detach().cpu().numpy()
    #     all_preds = torch.cat(preds).detach().cpu().numpy()
    #
    #     df = pd.DataFrame(all_probs, columns=['cls_0', 'cls_1', 'cls_2', 'cls_3'])
    #     df['ent'] = all_entropies
    #     df['max_probs'] = all_max_probs
    #     df['targets'] = all_targets
    #     df['preds'] = all_preds
    #     df.to_csv(f'/home/compu/jh/project/colon_compare/entropy{all_max_probs[0]}.csv')

    def on_epoch_end(self):
        self.train_acc.reset()
        self.val_acc.reset()
        self.test_acc.reset()
        self.train_acc_compare.reset()
        self.val_acc_compare.reset()
        self.test_acc_compare.reset()

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        self.scheduler = self.get_scheduler()
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler, 'monitor': 'val/loss'}

        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def get_scheduler(self):
        if self.hparams.scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.hparams.factor,
                patience=self.hparams.patience,
                verbose=True,
                eps=self.hparams.eps)
        elif self.hparams.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.hparams.t_max,
                eta_min=self.hparams.min_lr,
                last_epoch=-1)
        elif self.hparams.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.hparams.T_0,
                T_mult=1,
                eta_min=self.hparams.min_lr,
                last_epoch=-1)
        elif self.hparams.scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=200, gamma=0.1,
            )
        elif self.hparams.scheduler == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.95
            )
        # elif self.hparams.scheduler == 'MultiStepLR':
        #     scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #         )
        # elif self.hparams.scheduler == 'ConstantLR':
        #     scheduler = torch.optim.lr_scheduler.ConstantLR(
        #         )
        # elif self.hparams.scheduler == 'LinearLR':
        #     scheduler = torch.optim.lr_scheduler.LinearLR(
        #         )
        # elif self.hparams.scheduler == 'ChainedScheduler':
        #     scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        #         )
        # elif self.hparams.scheduler == 'SequentialLR':
        #     scheduler = torch.optim.lr_scheduler.SequentialLR(
        #         )
        # elif self.hparams.scheduler == 'CyclicLR':
        #     scheduler = torch.optim.lr_scheduler.CyclicLR(
        #         self.optimizer, base_lr=1e-5, max_lr=1e-2, step_size_up=5,mode="exp_range", gamma=0.95
        #     )
        # elif self.hparams.scheduler == 'OneCycleLR':
        #     scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #         self.optimizer, max_lr=1e-2,
        #     )

        return scheduler
