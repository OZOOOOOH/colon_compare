from typing import Any, List
import torch
import torch.nn as nn
import timm
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.functional import f1_score, cohen_kappa, accuracy
import random
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from src.datamodules.gastric_datamodule import CustomDataset
from src.utils import vote_results, get_shuffled_label
import copy
from scipy.stats import entropy
import operator


class GastricLitModule(LightningModule):
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
            key='ent',
            sampling='random',
            decide_by_total_probs=False,
            weighted_sum=False
    ):
        super(GastricLitModule, self).__init__()
        self.save_hyperparameters(logger=False)

        self.model = timm.create_model(self.hparams.name, pretrained=self.hparams.pretrained, num_classes=4)
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
        # self.fc_layer = nn.Sequential(
        #     nn.Linear(self.model.head.in_features, 512),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )
        # self.discriminator_layer3 = nn.Sequential(
        #     nn.Linear(512 * 2, 512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(256, 128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(128, 3),
        # )

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

    def forward(self, x):  # 4 classification
        return self.discriminator_layer1(self.model.forward_features(x.float()))

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

    def get_convinced(self, x, dataloader):
        convinced = []
        for img, label in dataloader:
            img = img.type_as(x)
            logits = self.forward(img)
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            max_probs = torch.max(probs, 1)

            escape = False
            for i, v in enumerate(max_probs.values):
                if v > 0.9 and preds[i] == label[0]:
                    convinced.append(img[i])
                    if len(convinced) == self.hparams.num_sample:
                        escape = True
                        break
            if escape:
                break
        return convinced

    def bring_random_trained_data(self, x):
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
                             'label': random_10_train_label0
                             })
        df_1 = pd.DataFrame({'path': random_10_train_path1,
                             'label': random_10_train_label1
                             })
        df_2 = pd.DataFrame({'path': random_10_train_path2,
                             'label': random_10_train_label2
                             })
        df_3 = pd.DataFrame({'path': random_10_train_path3,
                             'label': random_10_train_label3
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
        imgs_0, labels_0 = next(iter(dataloader_0))
        imgs_1, labels_1 = next(iter(dataloader_1))
        imgs_2, labels_2 = next(iter(dataloader_2))
        imgs_3, labels_3 = next(iter(dataloader_3))
        # bring data from train loader and convert to tensor
        imgs_0 = imgs_0.type_as(x)
        imgs_1 = imgs_1.type_as(x)
        imgs_2 = imgs_2.type_as(x)
        imgs_3 = imgs_3.type_as(x)

        return imgs_0, imgs_1, imgs_2, imgs_3

    def bring_convinced_trained_data(self, x):
        # bring convinced data
        self.trainer.datamodule.setup()
        train_img_path = self.trainer.datamodule.train_dataloader().dataset.image_id
        train_img_labels = self.trainer.datamodule.train_dataloader().dataset.labels

        idx_label0 = np.where(train_img_labels == 0)[0]
        idx_label1 = np.where(train_img_labels == 1)[0]
        idx_label2 = np.where(train_img_labels == 2)[0]
        idx_label3 = np.where(train_img_labels == 3)[0]

        train_path0 = train_img_path[idx_label0]
        train_path1 = train_img_path[idx_label1]
        train_path2 = train_img_path[idx_label2]
        train_path3 = train_img_path[idx_label3]

        train_label0 = train_img_labels[idx_label0]
        train_label1 = train_img_labels[idx_label1]
        train_label2 = train_img_labels[idx_label2]
        train_label3 = train_img_labels[idx_label3]

        df_0 = pd.DataFrame({'path': train_path0,
                             'label': train_label0
                             })
        df_1 = pd.DataFrame({'path': train_path1,
                             'label': train_label1
                             })
        df_2 = pd.DataFrame({'path': train_path2,
                             'label': train_label2
                             })
        df_3 = pd.DataFrame({'path': train_path3,
                             'label': train_label3
                             })

        dataloader_0 = DataLoader(
            CustomDataset(df_0, self.trainer.datamodule.train_transform),
            batch_size=16,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            shuffle=True,
        )
        dataloader_1 = DataLoader(
            CustomDataset(df_1, self.trainer.datamodule.train_transform),
            batch_size=16,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            shuffle=True,
        )
        dataloader_2 = DataLoader(
            CustomDataset(df_2, self.trainer.datamodule.train_transform),
            batch_size=16,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            shuffle=True,
        )
        dataloader_3 = DataLoader(
            CustomDataset(df_3, self.trainer.datamodule.train_transform),
            batch_size=16,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            shuffle=True,
        )
        return self.get_convinced(x, dataloader_0), self.get_convinced(x, dataloader_1), \
               self.get_convinced(x, dataloader_2), self.get_convinced(x, dataloader_3)

    def compare_test_with_trained(self, idx, features, imgs):
        # compare trained and test labels
        result_list = []
        for img in imgs:
            trained_features = self.model.forward_features(img.unsqueeze(0).float())
            concat_test_with_trained_features = torch.cat((features[idx].unsqueeze(0), trained_features), dim=1)
            logits_compare = self.discriminator_layer2(concat_test_with_trained_features)
            pred = torch.argmax(logits_compare, dim=1)
            result_list.append(pred)
        return result_list

    def compare_test_with_trained2(self, idx, features, imgs):
        # compare trained and test labels
        result_list = []
        for img in imgs:
            trained_features = self.model.forward_features(img.unsqueeze(0).float())
            concat_test_with_trained_features = torch.cat(
                (self.fc_layer(features[idx].unsqueeze(0)), self.fc_layer(trained_features)), dim=1)
            logits_compare = self.discriminator_layer3(concat_test_with_trained_features)
            pred = torch.argmax(logits_compare, dim=1)
            result_list.append(pred)
        return result_list

    def predict_using_voting(self, entopy_4cls, max_probs_4cls, features, total_imgs, probs_4cls, preds_4cls, y):
        # sourcery no-metrics

        imgs_0, imgs_1, imgs_2, imgs_3 = total_imgs
        cnt_correct_diff = 0
        ops = {'ent': operator.gt, 'prob': operator.lt}
        # gt: > , lt: <
        threshold_key = entopy_4cls if self.hparams.key == 'ent' else max_probs_4cls.values
        for idx, value in enumerate(threshold_key):
            if ops[self.hparams.key](value, self.hparams.threshold):
                result_0 = torch.cat(self.compare_test_with_trained(idx, features, imgs_0), dim=0)
                result_1 = torch.cat(self.compare_test_with_trained(idx, features, imgs_1), dim=0)
                result_2 = torch.cat(self.compare_test_with_trained(idx, features, imgs_2), dim=0)
                result_3 = torch.cat(self.compare_test_with_trained(idx, features, imgs_3), dim=0)
                # compare train imgs with test imgs // batch size
                vote_cnt = vote_results(result_0, result_1, result_2, result_3)
                total_score = sum(vote_cnt)
                prob_vote = np.array([i / total_score for i in vote_cnt])

                add_probs_4cls_vote = (probs_4cls[idx].detach().cpu().numpy() + prob_vote) / 2
                # divide 2 (1+1)
                vote_cls = np.argmax(add_probs_4cls_vote) if self.hparams.decide_by_total_probs else np.argmax(vote_cnt)

                e_4cls, e_vote = (
                    entropy(probs_4cls[idx].detach().cpu().numpy()),
                    entropy(prob_vote)) if self.hparams.weighted_sum else (
                    None, None)

                if e_4cls is not None and e_4cls > e_vote:
                    w_4cls_prob = np.exp(-e_4cls) / (np.exp(-e_4cls) + np.exp(-e_vote))
                    w_vot_prob = np.exp(-e_vote) / (np.exp(-e_4cls) + np.exp(-e_vote))

                    voting_classify = probs_4cls[
                                          idx].detach().cpu().numpy() * w_4cls_prob + prob_vote * w_vot_prob
                    vote_cls = np.argmax(voting_classify)

                # class based on voting
                if preds_4cls[idx].detach().cpu().item() != vote_cls:
                    if vote_cls == y[idx]:
                        cnt_correct_diff += 1
                    print()
                    print(f'True label: {y[idx]}')
                    print(f'Predict label: {preds_4cls[idx]}')
                    print(f'vote_cnt_0:{vote_cnt[0]}')
                    print(f'vote_cnt_1:{vote_cnt[1]}')
                    print(f'vote_cnt_2:{vote_cnt[2]}')
                    print(f'vote_cnt_3:{vote_cnt[3]}')
                    print(f'vote_cls:{vote_cls}')
                    print()
                preds_4cls[idx] = torch.Tensor([vote_cls]).type_as(y)
        return cnt_correct_diff, preds_4cls

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

    def step_test0(self, batch):  # only classification
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, logits, preds, y

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

        return loss, logits_4cls, logits_compare, preds_4cls, preds_compare, comparison, y, shuffle_y

    def step_test2(self, batch):  # random sampling or convinced sampling or weighted_sum or thresholding or
        x, y = batch
        features = self.model.forward_features(x.float())
        logits_4cls = self.discriminator_layer1(features)
        loss_4cls = self.criterion(logits_4cls, y)
        preds_4cls = torch.argmax(logits_4cls, dim=1)
        probs_4cls = torch.softmax(logits_4cls, dim=1)
        max_probs_4cls = torch.max(probs_4cls, 1)
        origin_preds_4cls = copy.deepcopy(preds_4cls)
        entropy_4cls = list(map(lambda i: entropy(i), probs_4cls.detach().cpu().numpy()))
        imgs_0, imgs_1, imgs_2, imgs_3 = self.bring_random_trained_data(
            x) if self.hparams.sampling == 'random' else self.bring_convinced_trained_data(x)

        total_imgs = [imgs_0, imgs_1, imgs_2, imgs_3]
        cnt_correct_diff, new_preds_4cls = self.predict_using_voting(entropy_4cls, max_probs_4cls, features, total_imgs,
                                                                     probs_4cls, preds_4cls, y)
        cnt_diff = sum(x != y for x, y in zip(origin_preds_4cls, new_preds_4cls))
        # print(f'length of preds : {len(origin_preds_4cls)} // The number of changed : {cnt_diff}')

        # losses = loss + loss_compare * self.hparams.loss_weight
        loss = loss_4cls

        return loss, logits_4cls, origin_preds_4cls, new_preds_4cls, y, cnt_diff, cnt_correct_diff

    def training_step(self, batch, batch_idx):
        # loss, logits, preds, targets = self.step_test0(batch)
        loss, logits_4cls, logits_compare, preds_4cls, preds_compare, comparison, targets, shuffle_y = self.step_test1(
            batch)
        # loss, preds, targets, preds_compare, comparison = self.step_after_concat(batch)
        # acc = self.train_acc(preds=preds, target=targets)
        acc = self.train_acc(preds=preds_4cls, target=targets)
        acc_compare = self.train_acc_compare(preds=preds_compare, target=comparison)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc_compare", acc_compare, on_step=True, on_epoch=True, prog_bar=True)
        self.log("LearningRate", self.optimizer.param_groups[0]['lr'])

        # return {"loss": loss, "acc": acc, "preds": preds, "targets": targets}
        return {"loss": loss, "acc": acc, "preds": preds_4cls, "targets": targets, "acc_compare": acc_compare,
                "preds_compare": preds_compare, "comparison": comparison}
        # return {"loss": loss, "acc": acc, "preds": preds, "targets": targets, "acc_compare": acc_compare,
        #         "preds_compare": preds_compare, "comparison": comparison}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val/loss"])

    def validation_step(self, batch, batch_idx):

        # loss, logits, preds, targets = self.step_test0(batch)
        loss, logits_4cls, logits_compare, preds_4cls, preds_compare, comparison, targets, shuffle_y = self.step_test1(
            batch)
        # loss, preds, targets, preds_compare, comparison = self.step_after_concat(batch)

        # acc = self.val_acc(preds, targets)
        acc = self.val_acc(preds_4cls, targets)
        acc_compare = self.val_acc_compare(preds=preds_compare, target=comparison)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_compare", acc_compare, on_step=False, on_epoch=True, prog_bar=True)
        # return {"loss": loss, "acc": acc, "preds": preds, "targets": targets}
        # return {"loss": loss, "acc": acc, "preds": preds, "targets": targets, "acc_compare": preds_compare,
        #         "preds_compare": preds_compare, "comparison": comparison}
        return {"loss": loss, "acc": acc, "preds": preds_4cls, "targets": targets, "acc_compare": preds_compare,
                "preds_compare": preds_compare, "comparison": comparison}

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        acc = self.val_acc.compute()
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

        acc_compare = self.val_acc_compare.compute()
        self.val_acc_compare_best.update(acc_compare)
        self.log("val/acc_compare_best", self.val_acc_compare_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, logits_4cls, origin_preds_4cls, new_preds_4cls, targets, cnt_diff, cnt_correct_diff = self.step_test2(
            batch)
        # loss, logits, logits_compare, preds, preds_compare, comparison, targets, shuffle_targets = self.step_test1(
        #     batch)
        # loss, logits, preds, targets = self.step_test0(batch)
        # loss, preds, targets, preds_compare, comparison = self.step_after_concat(batch)

        origin_acc = self.test_acc(origin_preds_4cls, targets)
        new_acc = self.test_acc(new_preds_4cls, targets)
        # acc = self.test_acc(preds, targets)
        # acc_compare = self.test_acc_compare(preds_compare, comparison)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/origin_acc", origin_acc, on_step=False, on_epoch=True)
        self.log("test/new_acc", new_acc, on_step=False, on_epoch=True)
        # self.log("test/acc", acc, on_step=False, on_epoch=True)
        # self.log("test/acc_compare", acc_compare, on_step=False, on_epoch=True)
        # return {"loss": loss, "acc": acc, "preds": preds, "targets": targets}
        # return {"loss": loss, "acc": acc, "preds": preds, "targets": targets, "acc_compare": preds_compare,
        #         "preds_compare": preds_compare, "comparison": comparison}

        return {"loss": loss, "origin_acc": origin_acc, "new_acc": new_acc, "origin_preds": origin_preds_4cls,
                "new_preds_4cls": new_preds_4cls, "targets": targets, "cnt_diff": cnt_diff,
                "cnt_correct_diff": cnt_correct_diff}

    def test_epoch_end(self, outputs):
        # pass
        # preds = torch.cat([i['preds'] for i in outputs], 0)
        # targets = torch.cat([i['targets'] for i in outputs], 0)

        preds = torch.cat([i['new_preds_4cls'] for i in outputs], 0)
        targets = torch.cat([i['targets'] for i in outputs], 0)

        acc = accuracy(preds, targets, num_classes=4)
        f1score_macro = f1_score(preds, targets, num_classes=4, average='macro')
        qwkappa = cohen_kappa(preds, targets, num_classes=4, weights='quadratic')
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        self.log("test/f1_macro", f1score_macro, on_step=False, on_epoch=True)
        self.log("test/wqKappa", qwkappa, on_step=False, on_epoch=True)

        cnt_diff = sum(i['cnt_diff'] for i in outputs)
        cnt_correct_diff = sum(i['cnt_correct_diff'] for i in outputs)
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
        schedulers = {
            'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.hparams.factor,
                patience=self.hparams.patience,
                verbose=True,
                eps=self.hparams.eps),
            'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.hparams.t_max,
                eta_min=self.hparams.min_lr,
                last_epoch=-1),
            'CosineAnnealingWarmRestarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.hparams.T_0,
                T_mult=1,
                eta_min=self.hparams.min_lr,
                last_epoch=-1),
            'StepLR': torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=200, gamma=0.1),
            'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.95),

        }
        if self.hparams.scheduler not in schedulers:
            raise ValueError(f'Unknown scheduler: {self.hparams.scheduler}')

        return schedulers.get(self.hparams.scheduler, schedulers['ReduceLROnPlateau'])
