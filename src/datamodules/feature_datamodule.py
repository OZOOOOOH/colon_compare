from src.utils import bring_dataset_csv, bring_dataset_colontest2_csv, bring_gastirc_dataset_csv
# from torchsampler import ImbalancedDatasetSampler
import pandas as pd
from src.datamodules.colon_datamodule import ColonDataset, ColonDataModule
from src.datamodules.colon_all_datamodule import ColonAllDataModule
from src.datamodules.harvard_datamodule import HarvardDataModule, prepare_prostate_harvard_data, HarvardDataset
from src.datamodules.ubc_datamodule import UbcDataModule, UbcDataset, make_dataset
from src.datamodules.prostate_datamodule import ProstateDataModule, ProstateDataset
from src.datamodules.gastric_datamodule import GastricDataModule, GastricDataset
class ColonDataModule(ColonDataModule):
    def __init__(
            self,
            data_dir: str = "./",
            img_size: int = 256,
            num_workers: int = 0,
            batch_size: int = 16,
            pin_memory=False,
            drop_last=False,
            data_name='colon',
            data_ratio=1.0,
    ):
        super().__init__()

    def setup(self, stage=None):
        # test for the trained dataset
        if stage != "fit" and stage is not None:
            train_df, _ = bring_dataset_csv(datatype='COLON_MANUAL_512', stage=None)
            if self.hparams.data_ratio<1.0:
                train_df=train_df.groupby('class').apply(lambda x:x.sample(frac=self.hparams.data_ratio)).reset_index(drop=True)            
            self.test_dataset = ColonDataset(train_df, self.test_transform)
class ColonAllDataModule(ColonAllDataModule):
    def __init__(
            self,
            data_dir: str = "./",
            img_size: int = 256,
            num_workers: int = 0,
            batch_size: int = 16,
            pin_memory=False,
            drop_last=False,
            data_name='colon',
            data_ratio=1.0
    ):
        super().__init__()

    def setup(self, stage=None):
        # test for the trained dataset
        if stage != "fit" and stage is not None:
            train_df1, _ = bring_dataset_csv(datatype='COLON_MANUAL_512', stage=None)
            train_df2, _= bring_dataset_colontest2_csv(stage=None)
            self.test_dataset = ColonDataset(pd.concat([train_df1,train_df2],ignore_index=True), self.test_transform)

class HarvardDataModule(HarvardDataModule):
    def __init__(
            self,
            data_dir: str = "./",
            img_size: int = 256,
            num_workers: int = 0,
            batch_size: int = 16,
            pin_memory=False,
            drop_last=False,
            data_name='harvard',
            data_ratio=1.0
    ):
        super().__init__()

    def setup(self, stage=None):
        # test for the trained dataset
        if stage != "fit" and stage is not None:
            train_set, _ = prepare_prostate_harvard_data(stage="train")
            if self.hparams.data_ratio<1.0:
                train_set=pd.DataFrame(train_set, columns =['path','class']).groupby('class').apply(lambda x:x.sample(frac=self.hparams.data_ratio,random_state=42)).reset_index(drop=True)
                train_set=list(train_set.to_records(index=False))
            self.test_dataset = HarvardDataset(train_set, self.test_transform)
    
class UbcDataModule(UbcDataModule):
    def __init__(
            self,
            data_dir: str = "./",
            img_size: int = 256,
            num_workers: int = 8,
            batch_size: int = 16,
            pin_memory=False,
            drop_last=False,
            data_name='ubc',
            data_ratio=1.0
    ):
        super().__init__()

    def setup(self, stage=None):
        # test for the trained dataset
        if stage != "fit" and stage is not None:
            train_set, _ = make_dataset(stage="train")
            if self.hparams.data_ratio<1.0:
                train_set=pd.DataFrame(train_set, columns =['path','class']).groupby('class').apply(lambda x:x.sample(frac=self.hparams.data_ratio,random_state=42)).reset_index(drop=True)
                train_set=list(train_set.to_records(index=False))
            self.test_dataset = UbcDataset(train_set, self.test_transform)

class ProstateDataModule(ProstateDataModule):
    def __init__(
            self,
            data_dir: str = "./",
            img_size: int = 256,
            num_workers: int = 8,
            batch_size: int = 16,
            pin_memory=False,
            drop_last=False,
            data_name='prostate',
            data_ratio=1.0
    ):
        super().__init__()

    def setup(self, stage=None):
        # test for the trained dataset
        if stage != "fit" and stage is not None:
            train_set1, _ = prepare_prostate_harvard_data(stage="train")
            train_set2, _ = make_dataset(stage="train")
            train_set=train_set1+train_set2
            if self.hparams.data_ratio<1.0:
                train_set=pd.DataFrame(train_set, columns =['path','class']).groupby('class').apply(lambda x:x.sample(frac=self.hparams.data_ratio,random_state=42)).reset_index(drop=True)
                train_set=list(train_set.to_records(index=False))
            self.test_dataset = ProstateDataset(train_set, self.test_transform)

class GastricDataModule(GastricDataModule):
    def __init__(
            self,
            data_dir: str = "./",
            img_size: int = 256,
            num_workers: int = 8,
            batch_size: int = 16,
            pin_memory=False,
            drop_last=False,
            data_name='gastric',
            data_ratio=1.0
    ):
        super().__init__()

    def setup(self, stage=None):
        # test for the trained dataset
        if stage != "fit" and stage is not None:
            train_df, _ = bring_gastirc_dataset_csv(stage=None)
            if self.hparams.data_ratio<1.0:
                train_df=train_df.groupby('class').apply(lambda x:x.sample(frac=self.hparams.data_ratio,random_state=42)).reset_index(drop=True)
            self.test_dataset = GastricDataset(train_df, self.test_transform)