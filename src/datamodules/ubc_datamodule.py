from glob import glob
from sklearn.model_selection import train_test_split
from src.datamodules.colon_test2_datamodule import ColonTestDataset
from src.datamodules.colon_datamodule import ColonDataModule, ColonDataset
import pandas as pd


class UbcDataset(ColonTestDataset):
    pass


class UbcDataset_pd(ColonDataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.labels = df["label"].values


class UbcDataModule(ColonDataModule):
    """
    prostate_miccai_2019_patches_690_80_step05
    class 0: 1811
    class 2: 7037
    class 3: 11431
    class 4: 292
    1284 BN, 5852 grade 3, 9682 grade 4, and 248 grade 5

    """

    def __init__(
        self,
        data_dir: str = "./",
        img_size: int = 256,
        num_workers: int = 4,
        batch_size: int = 16,
        pin_memory=False,
        drop_last=False,
        data_name="ubc",
        data_ratio=1.0,
    ):
        super().__init__()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:

            train_set, valid_set = make_ubc_dataset(stage="train")
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

            self.train_dataset = UbcDataset(train_set, self.train_transform)
            self.valid_dataset = UbcDataset(valid_set, self.test_transform)

        else:
            test_set = make_ubc_dataset(stage="test")
            self.test_dataset = UbcDataset(test_set, self.test_transform)


def splitting(dataset):  # train val test 80/10/10
    train, rest = train_test_split(
        dataset, train_size=0.8, shuffle=False, random_state=42
    )
    valid, test = train_test_split(rest, test_size=0.5, shuffle=False, random_state=42)
    return train, valid, test


def make_ubc_dataset(stage="train"):
    base_path = "/home/compu/jh/data/prostate_miccai_2019_patches_690_80_step05/"
    files = glob(f"{base_path}/*/*.jpg")

    data_class0 = [
        data for data in files if int(data.split("_")[-1].split(".")[0]) == 0
    ]
    data_class2 = [
        data for data in files if int(data.split("_")[-1].split(".")[0]) == 2
    ]
    data_class3 = [
        data for data in files if int(data.split("_")[-1].split(".")[0]) == 3
    ]
    data_class4 = [
        data for data in files if int(data.split("_")[-1].split(".")[0]) == 4
    ]

    train_data0, validation_data0, test_data0 = splitting(data_class0)
    train_data2, validation_data2, test_data2 = splitting(data_class2)
    train_data3, validation_data3, test_data3 = splitting(data_class3)
    train_data4, validation_data4, test_data4 = splitting(data_class4)
    label_dict = {0: 0, 2: 1, 3: 2, 4: 3}
    if stage == "train":

        train_path = train_data0 + train_data2 + train_data3 + train_data4
        valid_path = (
            validation_data0 + validation_data2 + validation_data3 + validation_data4
        )

        train_label = [int(path.split(".")[0][-1]) for path in train_path]
        valid_label = [int(path.split(".")[0][-1]) for path in valid_path]

        train_label = [label_dict[k] for k in train_label]
        valid_label = [label_dict[k] for k in valid_label]

        train_set = list(zip(train_path, train_label))
        valid_set = list(zip(valid_path, valid_label))
        return train_set, valid_set
    else:
        test_path = test_data0 + test_data2 + test_data3 + test_data4
        test_label = [int(path.split(".")[0][-1]) for path in test_path]
        test_label = [label_dict[k] for k in test_label]

        test_set = list(zip(test_path, test_label))
    return test_set
