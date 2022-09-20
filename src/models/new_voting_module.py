import torch
import numpy as np
import copy
from scipy.stats import entropy
import operator
from src.utils import get_confmat, tensor2np, bring_kmeans_trained_feature
import wandb
from collections import Counter
from src.models.voting_module import VotingLitModule


class NewVotingLitModule(VotingLitModule):
    def __init__(
        self,
        name="vit_large_r50_s32_384",
        pretrained=True,
        threshold=0.8,
        num_sample=30,
        key="ent",
        sampling="random",
        module_type="voting",
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.visited = [0, 0, 0, 0]
        self.duplicate = 0
        self.compare_class = [0, 0, 0, 0]
        self.compare_same = [0, 0, 0, 0]
        # self.cnt = SumMetric()

    def bring_trained_feature(self, mode):
        path = "/home/compu/jh/data/voting"
        name = f"{self.trainer.datamodule.__class__.__name__.lower()[:-10]}_{self.hparams.name}_1.0_seed_42.npy"
        features = np.load(f"{path}/features/{name}")
        preds = np.load(f"{path}/preds/{name}")
        targets = np.load(f"{path}/targets/{name}")
        if mode == "random":
            random_idxs = [
                np.random.choice(
                    np.where((preds == i) & (targets == i))[0],
                    self.hparams.num_sample,
                    replace=True,
                )
                for i in range(4)
            ]
        elif mode == "trust":
            max_probs = np.load(f"{path}/max_probs/{name}")
            random_idxs = [
                np.random.choice(
                    np.where((preds == i) & (max_probs > 0.9) & (targets == i))[0],
                    self.hparams.num_sample,
                    replace=True,
                )
                for i in range(4)
            ]
        elif mode == "kmeans":
            return bring_kmeans_trained_feature(features, targets, preds)

        return [features[random_idxs[i]] for i in range(4)]

    def compare_until_same(self, feature, trained_features, pred, origin_pred):
        """
        Compare until the result is "the same"

        This method is implemented by using recursion

        feature: A
        trained_features_by_class: B

        1) compare A with B
        2) if the result satisfies the condition, the function is ended
        3) else compare A with B'

        repeat 2), 3) until the condition is satisfied

        condition: the most highest value is 'same'

        B' --> change by the comparison

        ex) if prediction is 2, compare A with B which class is 2,
            if the comparison result is 1 --> end

            if prediction is 2, the comparison result is 2
            --> compare A with B' which class is 1
            ...
            --> if the comparison result is 1 --> end

        the main problem of this method is that infinite loops can occur

        ex) The prediction is 3 and the comparison result is 2,
            but if the result of comparison with B' which class is 2 is 0
            the infinite loop will occur

        so in that case, we will break the infinite loop and consider the class with the highest
        number of '1' as a prediction
        """
        assert 0 <= pred <= 3, "The prediction is out of range."

        results = self.compare_test_with_trained(
            feature, trained_features[pred]
        ).tolist()
        if not self.visited[pred]:
            self.visited[pred] = 1
        else:
            self.duplicate += 1
            for idx, v in enumerate(self.compare_class):
                r = self.compare_test_with_trained(
                    feature, trained_features[idx]
                ).tolist()
                top = sorted(Counter(r).most_common(3), key=lambda x: (-x[1], x[0]))
                if len(top) == 1:
                    if top[0][0] == 1:
                        self.compare_class[idx] = ("=", top[0][1])
                    elif top[0][0] == 0:
                        self.compare_class[idx] = ("<", top[0][1])
                    elif top[0][0] == 2:
                        self.compare_class[idx] = (">", top[0][1])
                elif len(top) >= 2 and top[0][1] > top[1][1]:
                    if top[0][0] == 1:
                        self.compare_class[idx] = ("=", top[0][1])
                    elif top[0][0] == 0:
                        self.compare_class[idx] = ("<", top[0][1])
                    elif top[0][0] == 2:
                        self.compare_class[idx] = (">", top[0][1])

                top = dict(top)
                if 1 in top:
                    self.compare_same[idx] = top[1]

            print("Something Wrong happened")
            print(self.compare_class)
            print(self.compare_same)

            return torch.tensor(np.argmax(self.compare_same)).type_as(pred)

        top = sorted(Counter(results).most_common(3), key=lambda x: (-x[1], x[0]))

        if (len(top) == 1 and top[0][0] == 1) or (
            len(top) >= 2 and top[0][1] != top[1][1] and top[0][0] == 1
        ):  # end condition
            self.compare_class[pred] = "="
            return pred

        elif len(top) == 3 and top[0][1] == top[1][1] == top[2][1]:
            print("Case 1 occurred")
            if pred == 1:

                return self.compare_until_same(
                    feature, trained_features, pred + 1, origin_pred
                )
            elif pred == 2:
                return self.compare_until_same(
                    feature, trained_features, pred - 1, origin_pred
                )

        elif len(top) == 2 and top[0][1] == top[1][1] and 1 in [top[0][0], top[1][0]]:

            if 0 in [top[0][0], top[1][0]]:
                # >,=
                print("Case 2-1 occurred")
                return self.compare_until_same(
                    feature, trained_features, pred + 1, origin_pred
                )
            elif 2 in [top[0][0], top[1][0]]:
                # <,=
                print("Case 2-2 occurred")
                return self.compare_until_same(
                    feature, trained_features, pred - 1, origin_pred
                )

        elif (
            len(top) >= 2
            and top[0][1] == top[1][1]
            and 0 in [top[0][0], top[1][0]]
            and 2 in [top[0][0], top[1][0]]
        ):
            # <,>
            print("Case 2-3 occurred")
            return self.compare_until_same(
                feature, trained_features, pred + 1, origin_pred
            )
            return self.compare_until_same(
                feature, trained_features, pred - 1, origin_pred
            )

        elif top[0][0] == 0:
            # >
            print("Case 4-1 occurred")
            self.compare_class[pred] = "<"
            return self.compare_until_same(
                feature, trained_features, pred + 1, origin_pred
            )

        elif top[0][0] == 2:
            # <
            print("Case 4-2 occurred")
            self.compare_class[pred] = ">"
            return self.compare_until_same(
                feature, trained_features, pred - 1, origin_pred
            )
        else:
            print("idk Case occurred")
            print(top)

    def voting(
        self,
        entropies,
        max_probs,
        features,
        trained_features,
        preds,
        y,
    ):
        """
        voting function

        1) get the threshold key value (entropy or probability)
        2) Check whether the threshold condition is satisfied
            (a) if the key value is entropy, then the condition is "value > threshold"
            (b) if the key value is probability, then the condition is "value < threshold"
        3) If satisfied, go to compare_until_same
        """
        cnt_correct_diff = 0
        ops = {"ent": operator.gt, "prob": operator.lt}
        # gt: > , lt: <
        threshold_key = entropies if self.hparams.key == "ent" else max_probs.values
        # get key data by key
        for idx, value in enumerate(threshold_key):
            if ops[self.hparams.key](value, self.hparams.threshold):
                # if key=='ent' --> value > threshold
                # if key=='prob' --> value < threshold
                self.visited = [0] * 4
                self.compare_class = [0] * 4
                self.compare_same = [0] * 4
                result = self.compare_until_same(
                    features[idx],
                    trained_features,
                    preds[idx],
                    preds[idx],
                )
                print(y[idx], preds[idx])
                if tensor2np(preds[idx]) != tensor2np(result):
                    preds[idx] = result
                    if tensor2np(result) == tensor2np(y[idx]):
                        cnt_correct_diff += 1

        return cnt_correct_diff, preds

    def step(self, batch):
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
        cnt_correct_diff, new_preds_4cls = self.voting(
            entropy_4cls, max_probs_4cls, features, trained_features, preds_4cls, y
        )
        cnt_diff = sum(x != y for x, y in zip(origin_preds_4cls, new_preds_4cls))
        # print(f'length of preds : {len(origin_preds_4cls)} // The number of changed : {cnt_diff}')
        return (
            loss_4cls,
            origin_preds_4cls,
            new_preds_4cls,
            y,
            cnt_diff,
            cnt_correct_diff,
        )

    def test_step(self, batch, batch_idx):

        (
            loss,
            origin_preds_4cls,
            new_preds_4cls,
            target_4cls,
            cnt_diff,
            cnt_correct_diff,
        ) = self.step(batch)
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
        self.log(
            "test/cnt_diff", cnt_diff, on_epoch=True, on_step=False, reduce_fx="sum"
        )
        self.log(
            "test/cnt_correct_diff",
            cnt_correct_diff,
            on_epoch=True,
            on_step=False,
            reduce_fx="sum",
        )
        self.test_acc.reset()
        self.confusion_matrix.reset()
        self.f1_score.reset()
        self.cohen_kappa.reset()
