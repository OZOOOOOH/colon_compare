from src.datamodules.ubc_datamodule import make_ubc_dataset
from src.datamodules.harvard_datamodule import (
    HarvardDataset,
    prepare_prostate_harvard_data,
)
from src.datamodules.gastric_datamodule import GastricDataset
from src.datamodules.colon_datamodule import ColonDataModule
import pandas as pd


class ProstateDataset(HarvardDataset):
    pass


class ProstateDataset_pd(GastricDataset):
    pass


class ProstateDataModule(ColonDataModule):
    """
    - (A) havard Dataset
    - (B) ubc Dataset

    Prostate = (A) + (B)
    """

    def __init__(
        self,
        data_dir: str = "./",
        img_size: int = 256,
        num_workers: int = 4,
        batch_size: int = 16,
        pin_memory=False,
        drop_last=False,
        data_name="harvard",
        data_ratio=1.0,
    ):
        super().__init__()

    def setup(self, stage=None):
        # Assign train/val/test datasets for use in dataloaders
        if stage == "fit" or stage is None:

            train_set1, valid_set1 = prepare_prostate_harvard_data(stage="train")
            train_set2, valid_set2 = make_ubc_dataset(stage="train")
            train_set = train_set1 + train_set2
            if self.hparams.data_ratio < 1.0:
                train_set = (
                    pd.DataFrame(train_set, columns=["path", "class"])
                    .groupby("class")
                    .apply(
                        lambda x: x.sample(
                            frac=self.hparams.data_ratio, random_state=42
                        )
                    )
                    .reset_index(drop=True)
                )
                train_set = list(train_set.to_records(index=False))

            self.train_dataset = ProstateDataset(train_set, self.train_transform)
            self.valid_dataset = ProstateDataset(
                valid_set1 + valid_set2, self.test_transform
            )

        else:
            test_set1 = prepare_prostate_harvard_data(stage="test")
            test_set2 = make_ubc_dataset(stage="test")

            self.test_dataset = ProstateDataset(
                test_set1 + test_set2, self.test_transform
            )
