from glob import glob
# from torchsampler import ImbalancedDatasetSampler
from src.datamodules.colon_datamodule import ColonDataset, ColonDataModule
from src.datamodules.colon_test2_datamodule import ColonTestDataset
import pandas as pd
class HarvardDataset(ColonTestDataset):
    pass
class HarvardDataset_pd(ColonDataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.labels = df["label"].values

class HarvardDataModule(ColonDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        img_size: int = 256,
        num_workers: int = 4,
        batch_size: int = 16,
        pin_memory=False,
        drop_last=False,
        data_name='harvard',
        data_ratio=1.0
    ):
        super().__init__()

    def setup(self, stage=None):
        # Assign train/val/test datasets for use in dataloaders
        if stage == "fit" or stage is None:

            train_set, valid_set = prepare_prostate_harvard_data(stage="train")
            if self.hparams.data_ratio<1.0:
                train_set=pd.DataFrame(train_set, columns =['path','class']).groupby('class').apply(lambda x:x.sample(frac=self.hparams.data_ratio,random_state=42)).reset_index(drop=True)
                train_set=list(train_set.to_records(index=False))

            self.train_dataset = HarvardDataset(train_set, self.train_transform)
            self.valid_dataset = HarvardDataset(valid_set, self.test_transform)

        else:
            test_set = prepare_prostate_harvard_data(stage="test")
            self.test_dataset = HarvardDataset(test_set, self.test_transform)

def prepare_prostate_harvard_data(stage="train"):
    def load_data_info(pathname):
        file_list = glob(pathname)

        label_list = [int(file_path.split("_")[-1].split(".")[0]) for file_path in file_list]

        return list(zip(file_list, label_list))

    data_root_dir = "/home/compu/jh/data/Harvard_ReplicationData_Prostate"
    data_root_dir_train = f"{data_root_dir}/patches_train_750_v0/"
    data_root_dir_valid = f"{data_root_dir}/patches_validation_750_v0/"
    data_root_dir_test = f"{data_root_dir}/patches_test_750_v0/"

    if stage == "train":
        train_set_111 = load_data_info(f"{data_root_dir_train}/ZT111*/*.jpg")
        train_set_199 = load_data_info(f"{data_root_dir_train}/ZT199*/*.jpg")
        train_set_204 = load_data_info(f"{data_root_dir_train}/ZT204*/*.jpg")
        valid_set = load_data_info(f"{data_root_dir_valid}/ZT76*/*.jpg")
        train_set = train_set_111 + train_set_199 + train_set_204

        return train_set, valid_set

    elif stage == "test":
        test_set = load_data_info(f"{data_root_dir_test}/patho_1/*/*.jpg")
        return test_set