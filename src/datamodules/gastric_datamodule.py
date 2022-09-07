from src.utils import bring_gastirc_dataset_csv
# from torchsampler import ImbalancedDatasetSampler
from src.datamodules.colon_datamodule import ColonDataset, ColonDataModule

class GastricDataset(ColonDataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.labels = df["label"].values

class GastricDataModule(ColonDataModule):
    def __init__(
            self,
            data_dir: str = "./",
            img_size: int = 256,
            num_workers: int = 4,
            batch_size: int = 16,
            pin_memory=False,
            drop_last=False,
            data_name='gastric',
            data_ratio=1.0
    ):
        super().__init__()
    def setup(self, stage=None):
        # Assign train/val/test datasets for use in dataloaders
        if stage == "fit" or stage is None:
            # Random train-validation split
            train_df, valid_df = bring_gastirc_dataset_csv(stage=None)
            # Train dataset
            if self.hparams.data_ratio<1.0:
                train_df=train_df.groupby('class').apply(lambda x:x.sample(frac=self.hparams.data_ratio,random_state=42)).reset_index(drop=True)
            self.train_dataset = GastricDataset(train_df, self.train_transform)
            # Validation dataset
            self.valid_dataset = GastricDataset(valid_df, self.test_transform)
            # Test dataset
        else:
            test_df = bring_gastirc_dataset_csv(stage='test')
            self.test_dataset = GastricDataset(test_df, self.test_transform)