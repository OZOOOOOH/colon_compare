import itertools
from typing import Any, List
import torch
import torch.nn as nn
import timm
from pytorch_lightning import LightningModule
from torchmetrics import ConfusionMatrix, F1Score, CohenKappa, Accuracy
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd

from src.datamodules.ubc_datamodule import UbcDataset_pd
from src.datamodules.harvard_datamodule import HarvardDataset_pd
from src.datamodules.colon_datamodule import ColonDataset
from src.datamodules.gastric_datamodule import GastricDataset

from src.utils import vote_results, get_shuffled_label
import copy
from scipy.stats import entropy
import operator
from src.utils import vote_results, get_shuffled_label, get_confmat, tensor2np
import wandb


class VotingLitModule(LightningModule):
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
            module_type="voting",
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
            self.model.forward_head(self.model.forward_features(x.float()), pre_logits=True)
        )

    def get_features(self, x):
        # get features from model
        features = self.model.forward_features(x.float())
        features = self.model.forward_head(features, pre_logits=True)
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

    def check_which_dataset(self):
        dataset_name = str(self.trainer.datamodule.__class__).lower()
        if "ubc" in dataset_name:
            return "ubc"
        elif "harvard" in dataset_name:
            return "harvard"
        elif "colon" in dataset_name:
            return "colon"
        elif "gastric" in dataset_name:
            return "gastric"
        else:
            raise ValueError("Dataset name is not correct")

    def get_trained_dataset(self):
        data_type = self.check_which_dataset()
        if data_type in ["colon", "gastric"]:
            paths = self.trainer.datamodule.train_dataloader().dataset.image_id
            labels = self.trainer.datamodule.train_dataloader().dataset.labels
        elif data_type in ["harvard", "ubc"]:
            paths = np.array(
                [
                    path
                    for path, label in self.trainer.datamodule.train_dataloader().dataset.pair_list
                ]
            )
            labels = np.array(
                [
                    label
                    for path, label in self.trainer.datamodule.train_dataloader().dataset.pair_list
                ]
            )

        return paths, labels, data_type

    def get_dataclass(self, data_type):
        if data_type == "colon":
            return ColonDataset
        elif data_type == "gastric":
            return GastricDataset
        elif data_type == "harvard":
            return HarvardDataset_pd
        elif data_type == "ubc":
            return UbcDataset_pd

    def bring_trained_feature(self, mode):
        path = '/home/compu/jh/data/voting/'
        name = f'{self.trainer.datamodule.__class__.__name__.lower()[:-10]}_{self.hparams.name}.npy'
        features = np.load(f'{path}/features/{name}')
        preds = np.load(f'{path}/preds/{name}')

        if mode == 'random':
            random_idxs = [np.random.choice(np.where(preds == i)[0], self.hparams.num_sample, replace=False) for i in
                           range(4)]
        elif mode == 'trust':
            max_probs = np.load(f'{path}/max_probs/{name}')
            random_idxs = [
                np.random.choice(np.where(np.logical_and(preds == i, max_probs > 0.9))[0], self.hparams.num_sample,
                                 replace=False) for i in range(4)]

        return [features[random_idxs[i]] for i in range(4)]

    def bring_random_trained_data(self, x):
        self.trainer.datamodule.setup()
        train_img_path, train_img_labels, data_type = self.get_trained_dataset()
        CustomDataset = self.get_dataclass(data_type)
        random_idx_labels = [
            np.random.choice(
                np.where(train_img_labels == i)[0], self.hparams.num_sample, replace=False
            )
            for i in range(4)
        ]
        random_10_train_paths = [train_img_path[random_idx_labels[i]] for i in range(4)]
        random_10_train_labels = [train_img_labels[random_idx_labels[i]] for i in range(4)]

        df = [
            pd.DataFrame(
                {
                    "path": random_10_train_paths[i],
                    "label": random_10_train_labels[i],
                    "class": random_10_train_labels[i],
                }
            )
            for i in range(4)
        ]

        dataloaders = [
            DataLoader(
                CustomDataset(df[i], self.trainer.datamodule.train_transform),
                batch_size=10,
                num_workers=0,
                pin_memory=False,
                drop_last=False,
                shuffle=False,
            )
            for i in range(4)
        ]

        imgs = []
        for i in range(4):
            img, _ = next(iter(dataloaders[i]))
            img = img.type_as(x)
            imgs.append(img)
        # bring data from train loader and convert to tensor

        return imgs

    def bring_convinced_trained_data(self, x):
        # bring convinced data
        self.trainer.datamodule.setup()
        train_img_path, train_img_labels, data_type = self.get_trained_dataset()
        CustomDataset = self.get_dataclass(data_type)

        idx_labels = [np.where(train_img_labels == i)[0] for i in range(4)]
        train_paths = [train_img_path[idx_labels[i]] for i in range(4)]
        train_labels = [train_img_labels[idx_labels[i]] for i in range(4)]
        df = [
            pd.DataFrame(
                {"path": train_paths[i], "label": train_labels[i], "class": train_labels[i]}
            )
            for i in range(4)
        ]
        dataloaders = [
            DataLoader(
                CustomDataset(df[i], self.trainer.datamodule.train_transform),
                batch_size=16,
                num_workers=0,
                pin_memory=False,
                drop_last=False,
                shuffle=True,
            )
            for i in range(4)
        ]

        return [self.get_convinced(x, dataloaders[i]) for i in range(4)]

    def compare_test_with_trained(self, idx, features, trained_features):
        # compare trained and test labels
        result_list = []
        for trained_feature in trained_features:
            concat_test_with_trained_features = torch.cat(
                (features[idx].unsqueeze(0), torch.from_numpy(trained_feature).type_as(features[idx]).unsqueeze(0)),
                dim=1
            )
            logits_compare = self.discriminator_layer2(concat_test_with_trained_features)
            preds_compare = torch.argmax(logits_compare, dim=1)
            result_list.append(preds_compare)
        return torch.cat(result_list)

    def predict_using_voting(
            self, entropy_4cls, max_probs_4cls, features, total_trained_features, probs_4cls, preds_4cls, y
    ):
        cnt_correct_diff = 0
        ops = {"ent": operator.gt, "prob": operator.lt}
        # gt: > , lt: <
        threshold_key = entropy_4cls if self.hparams.key == "ent" else max_probs_4cls.values
        # get key data by key
        for idx, value in enumerate(threshold_key):
            if ops[self.hparams.key](value, self.hparams.threshold):
                # if key=='ent' --> value > threshold
                # if key=='prob' --> value < threshold
                results = [
                    self.compare_test_with_trained(idx, features, total_trained_features[i])
                    for i in range(4)
                ]
                # compare train imgs with test imgs // batch size
                vote_score = vote_results(results)  # vote_score

                if self.hparams.weighted_sum:
                    total_score = sum(vote_score)
                    prob_vote = np.array(
                        [i / total_score for i in vote_score]
                    )  # make vote_socre to probability
                    vote_cls = self.get_vote_cls_by_weighted_sum(entropy_4cls, probs_4cls, idx, prob_vote)
                elif self.hparams.decide_by_total_probs:
                    total_score = sum(vote_score)
                    prob_vote = np.array(
                        [i / total_score for i in vote_score]
                    )  # make vote_socre to probability

                    vote_cls = self.get_vote_cls_by_total_probs(probs_4cls, idx, prob_vote)
                else:
                    vote_cls = np.argmax(vote_score)

                if tensor2np(preds_4cls)[idx] != vote_cls and vote_cls == y[idx]:
                    cnt_correct_diff += 1
                # This is for the counting that the vote_label is the same as the true label, but differnt as predict label
                preds_4cls[idx] = torch.Tensor([vote_cls]).type_as(y)
        return cnt_correct_diff, preds_4cls

    def get_vote_cls_by_total_probs(self, probs_4cls, idx, prob_vote):
        # add probability of vote and original probs_4cls
        # divide 2 (1+1)
        add_probs_4cls_vote = (tensor2np(probs_4cls[idx]) + prob_vote) / 2
        return np.argmax(add_probs_4cls_vote)

    def get_vote_cls_by_weighted_sum(self, entropy_4cls, probs_4cls, idx, prob_vote, vote_cnt):
        e_4cls, e_vote = entropy_4cls[idx], entropy(prob_vote)
        if e_4cls <= e_vote:
            return np.argmax(vote_cnt)
        w_4cls = np.exp(-e_4cls) / (np.exp(-e_4cls) + np.exp(-e_vote))
        w_vote = np.exp(-e_vote) / (np.exp(-e_4cls) + np.exp(-e_vote))
        voting_classify = tensor2np(probs_4cls[idx]) * w_4cls + prob_vote * w_vote
        return np.argmax(voting_classify)

    def step_voting(self, batch):
        x, y = batch
        features = self.get_features(x)
        logits_4cls = self.discriminator_layer1(features)
        loss_4cls = self.criterion(logits_4cls, y)
        preds_4cls = torch.argmax(logits_4cls, dim=1)
        probs_4cls = torch.softmax(logits_4cls, dim=1)
        max_probs_4cls = torch.max(probs_4cls, 1)
        origin_preds_4cls = copy.deepcopy(preds_4cls)
        entropy_4cls = list(map(lambda i: entropy(i), tensor2np(probs_4cls)))

        # total_trained_imgs = (
        #     self.bring_random_trained_data(x)
        #     if self.hparams.sampling == "random"
        #     else self.bring_convinced_trained_data(x)
        # )
        trained_features = self.bring_trained_feature(mode=self.hparams.sampling)
        cnt_correct_diff, new_preds_4cls = self.predict_using_voting(
            entropy_4cls, max_probs_4cls, features, trained_features, probs_4cls, preds_4cls, y
        )
        cnt_diff = sum(x != y for x, y in zip(origin_preds_4cls, new_preds_4cls))
        # print(f'length of preds : {len(origin_preds_4cls)} // The number of changed : {cnt_diff}')

        # losses = loss + loss_compare * self.hparams.loss_weight
        loss = loss_4cls

        return loss, origin_preds_4cls, new_preds_4cls, y, cnt_diff, cnt_correct_diff

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):

        (
            loss,
            origin_preds_4cls,
            new_preds_4cls,
            target_4cls,
            cnt_diff,
            cnt_correct_diff,
        ) = self.step_voting(batch)
        self.confusion_matrix(new_preds_4cls, target_4cls)
        self.f1_score(new_preds_4cls, target_4cls)
        self.cohen_kappa(new_preds_4cls, target_4cls)

        origin_acc = self.test_acc(origin_preds_4cls, target_4cls)
        new_acc = self.test_acc(new_preds_4cls, target_4cls)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/origin_acc", origin_acc, on_step=False, on_epoch=True)
        self.log("test/new_acc", new_acc, on_step=False, on_epoch=True)
        return {
            "loss": loss,
            "origin_acc": origin_acc,
            "new_acc": new_acc,
            "origin_preds": origin_preds_4cls,
            "new_preds_4cls": new_preds_4cls,
            "targets": target_4cls,
            "cnt_diff": cnt_diff,
            "cnt_correct_diff": cnt_correct_diff,
        }

    def test_epoch_end(self, outputs):

        cm = self.confusion_matrix.compute()
        f1 = self.f1_score.compute()
        qwk = self.cohen_kappa.compute()
        p = get_confmat(cm)
        self.logger.experiment.log({"test/conf_matrix": wandb.Image(p)})

        self.log("test/f1_macro", f1, on_step=False, on_epoch=True)
        self.log("test/wqKappa", qwk, on_step=False, on_epoch=True)

        cnt_diff = sum(i["cnt_diff"] for i in outputs)
        cnt_correct_diff = sum(i["cnt_correct_diff"] for i in outputs)
        cnt_diff = cnt_diff.sum()
        self.log("test/cnt_diff", cnt_diff, on_epoch=True, on_step=False, reduce_fx="sum")
        self.log(
            "test/cnt_correct_diff", cnt_correct_diff, on_epoch=True, on_step=False, reduce_fx="sum"
        )
        self.test_acc.reset()
        self.confusion_matrix.reset()
        self.f1_score.reset()
        self.cohen_kappa.reset()
