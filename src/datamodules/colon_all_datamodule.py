from src.utils import bring_dataset_csv, bring_dataset_colontest2_csv
import pandas as pd
# from torchsampler import ImbalancedDatasetSampler
from src.datamodules.colon_datamodule import ColonDataset, ColonDataModule
class ColonAllDataModule(ColonDataModule):
    def __init__(
            self,
            data_dir: str = "./",
            img_size: int = 256,
            num_workers: int = 0,
            batch_size: int = 16,
            pin_memory=False,
            drop_last=False,
            data_name='colon'

    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
    def setup(self, stage=None):
        # Assign train/val/test datasets for use in dataloaders
        if stage == "fit" or stage is None:
            # Random train-validation split
            train_df1, valid_df1 = bring_dataset_csv(datatype='COLON_MANUAL_512', stage=None)
            train_df2, valid_df2= bring_dataset_colontest2_csv(stage=None)
            # Train dataset
            self.train_dataset = ColonDataset(pd.concat([train_df1,train_df2],ignore_index=True), self.train_transform)
            # Validation dataset
            self.valid_dataset = ColonDataset(pd.concat([valid_df1,valid_df2],ignore_index=True), self.test_transform)
            # Test dataset
        else:
            test_df1 = bring_dataset_csv(datatype='COLON_MANUAL_512', stage='test')
            test_df2 = bring_dataset_colontest2_csv(stage='test')
            self.test_dataset = ColonDataset(pd.concat([test_df1,test_df2]), self.test_transform)