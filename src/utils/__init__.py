import logging
import warnings
from typing import List, Sequence
import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
import pandas as pd
import math
from torch.optim.lr_scheduler import _LRScheduler
import random
import torch
from functools import reduce
from torch.nn.functional import pairwise_distance
from torch.nn import _reduction as _Reduction
from typing import Callable, Optional
from torch import nn


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
            "debug",
            "info",
            "warning",
            "error",
            "exception",
            "fatal",
            "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)


def extras(config: DictConfig) -> None:
    """Applies optional utilities, controlled by config flags.

    Utilities:
    - Ignoring python warnings
    - Rich config printing
    """

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library if <config.print_config=True>
    if config.get("print_config"):
        log.info("Printing config tree with Rich! <config.print_config=True>")
        print_config(config, resolve=True)


@rank_zero_only
def print_config(
        config: DictConfig,
        print_order: Sequence[str] = (
                "datamodule",
                "model",
                "callbacks",
                "logger",
                "trainer",
        ),
        resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    quee = []

    for field in print_order:
        quee.append(field) if field in config else log.info(f"Field '{field}' not found in config")

    for field in config:
        if field not in quee:
            quee.append(field)

    for field in quee:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = config[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as file:
        rich.print(tree, file=file)


@rank_zero_only
def log_hyperparameters(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List[pl.loggers.LightningLoggerBase],
) -> None:  # sourcery skip: merge-dict-assign
    """Controls which config parts are saved by Lightning loggers.

    Additionaly saves:
    - number of model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["model"] = config["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = config["datamodule"]
    hparams["trainer"] = config["trainer"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def finish(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb
            wandb.finish()


def bring_dataset_csv(datatype, stage=None):
    # Directories
    path = f"/home/compu/jh/data/colon_tma/{datatype}/"

    if stage != "fit" and stage is not None:
        return pd.read_csv(path + "test.csv")
    df_train = pd.read_csv(path + "train.csv")
    df_val = pd.read_csv(path + "valid.csv")
    return df_train, df_val


def bring_gastirc_dataset_csv(stage=None):
    # Directories
    path = "/home/compu/jh/data/gastric/"

    if stage != "fit" and stage is not None:
        return pd.read_csv(f"{path}class4_step10_ds_test.csv")
    df_train = pd.read_csv(f"{path}class4_step05_ds_train.csv")
    df_val = pd.read_csv(f"{path}class4_step10_ds_valid.csv")
    return df_train, df_val


def get_shuffled_label(x, y):
    pair = list(enumerate(list(pair) for pair in zip(x, y)))
    pair = random.sample(pair, len(pair))
    indices, pair = zip(*pair)
    indices = list(indices)
    pair = list(pair)
    shuffle_y = [i[1] for i in pair]
    shuffle_y = torch.stack(shuffle_y, dim=0)

    return indices, shuffle_y


def vote_results(result_0, result_1, result_2, result_3):
    vote_cnt_0 = 0
    vote_cnt_1 = 0
    vote_cnt_2 = 0
    vote_cnt_3 = 0
    vote_cnt_else = 0
    for i in result_0:
        if i == 0:
            vote_cnt_1 += 1
            vote_cnt_2 += 1
            vote_cnt_3 += 1
        elif i == 1:
            vote_cnt_0 += 1
        else:
            vote_cnt_else += 1
    print(f'In result_0: {vote_cnt_0} + {vote_cnt_1} + {vote_cnt_2} + {vote_cnt_3} + {vote_cnt_else}')
    for i in result_1:
        if i == 0:
            vote_cnt_2 += 1
            vote_cnt_3 += 1
        elif i == 1:
            vote_cnt_1 += 1
        elif i == 2:
            vote_cnt_0 += 1
    print(f'In result_1: {vote_cnt_0} + {vote_cnt_1} + {vote_cnt_2} + {vote_cnt_3} + {vote_cnt_else}')
    for i in result_2:
        if i == 0:
            vote_cnt_3 += 1
        elif i == 1:
            vote_cnt_2 += 1
        elif i == 2:
            vote_cnt_0 += 1
            vote_cnt_1 += 1
    print(f'In result_2: {vote_cnt_0} + {vote_cnt_1} + {vote_cnt_2} + {vote_cnt_3} + {vote_cnt_else}')
    for i in result_3:
        if i == 0:
            vote_cnt_else += 1
        elif i == 1:
            vote_cnt_3 += 1
        elif i == 2:
            vote_cnt_0 += 1
            vote_cnt_1 += 1
            vote_cnt_2 += 1
    print(f'In result_3: {vote_cnt_0} + {vote_cnt_1} + {vote_cnt_2} + {vote_cnt_3} + {vote_cnt_else}')
    return [vote_cnt_0, vote_cnt_1, vote_cnt_2, vote_cnt_3]


def dist_indexing(y, shuffle_y, y_idx_groupby, dist_matrix):
    indices = []
    for i, (yV, shuffleV) in enumerate(zip(y, shuffle_y)):
        if yV == shuffleV:
            indices.append(y_idx_groupby[yV][dist_matrix[i][y_idx_groupby[yV]].argmax()])
        elif yV > shuffleV:
            flatten_ = reduce(lambda a, b: a + b, y_idx_groupby[:yV])
            indices.append(flatten_[dist_matrix[i][flatten_].argmin()])
        else:
            flatten_ = reduce(lambda a, b: a + b, y_idx_groupby[yV + 1:])
            indices.append(flatten_[dist_matrix[i][flatten_].argmin()])
    return indices


def params_freeze(model):
    model.blocks[22:].requires_grad_(False)
    for name, param in model.named_parameters():
        param.requires_grad = 'head' in name
        param.requires_grad = 'norm' in name


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError(f"Expected positive integer T_up, but got {T_up}")
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (
                    1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        elif epoch >= self.T_0:
            if self.T_mult == 1:
                self.T_cur = epoch % self.T_0
                self.cycle = epoch // self.T_0
            else:
                n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                self.cycle = n
                self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                self.T_i = self.T_0 * self.T_mult ** (n)
        else:
            self.T_i = self.T_0
            self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def triplet_margin_with_distance_loss(
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        *,
        distance_function: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        margin: float = 1.0,
        swap: bool = False,
        reduction: str = "mean"
) -> torch.Tensor:
    r"""
    See :class:`~torch.nn.TripletMarginWithDistanceLoss` for details.
    """
    if torch.jit.is_scripting():
        raise NotImplementedError(
            "F.triplet_margin_with_distance_loss does not support JIT scripting: "
            "functions requiring Callables cannot be scripted."
        )

    distance_function = distance_function if distance_function is not None else pairwise_distance

    positive_dist = distance_function(anchor, positive)
    negative_dist = distance_function(anchor, negative)

    if swap:
        swap_dist = distance_function(positive, negative)
        negative_dist = torch.min(negative_dist, swap_dist)

    output = torch.clamp(positive_dist - negative_dist + margin, min=0.0)

    reduction_enum = _Reduction.get_enum(reduction)
    if reduction_enum == 1:
        return output.mean()
    elif reduction_enum == 2:
        return output.sum()
    else:
        return output


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)
        # This is same as " max( D(a,p)-D(a,n) + margin, 0 ) "


class TripletLossWithGL(nn.Module):
    """Triplet loss with hard positive/negative mining and Greater/Less.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(TripletLossWithGL, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask_eq = torch.mul(targets.expand(n, n).eq(targets.expand(n, n).t()), 1)  # =
        mask_gt = torch.mul(targets.expand(n, n).gt(targets.expand(n, n).t()), 2)  # <
        mask_lt = torch.mul(targets.expand(n, n).lt(targets.expand(n, n).t()), 3)  # >
        mask = mask_eq + mask_gt + mask_lt
        dist_ap, dist_an_g, dist_an_l = [], [], []
        # dist_anG: anchor negative greater, dist_anL: anchor negative less,
        for i in range(n):
            dist_ap.append(dist[i][mask[i] == 1].max().unsqueeze(0))
            dist_an_g.append(dist[i][mask[i] == 2].min().unsqueeze(0))
            dist_an_l.append(dist[i][mask[i] == 3].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an_g = torch.cat(dist_an_g)
        dist_an_l = torch.cat(dist_an_l)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an_g)
        return self.ranking_loss(abs(dist_an_g - dist_an_l), dist_ap, y)
        # This is same as " max( D(a,p)-|D(a,n>)-D(a,n<)| + margin, 0 ) "
