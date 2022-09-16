import albumentations as A
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from src.utils import bring_colon_dataset_csv
from cv2 import cv2


class ColonDataset(Dataset):
    def __init__(self, df, transform=None):
        self.image_id = df["path"].values
        self.labels = df["class"].values
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.image_id[idx]
        label = self.labels[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)
        return image["image"], label


class ColonDataModule(LightningDataModule):
    """
    There is 1 dataset of colon

    - (A) is the datasets that has Training Validation Testing I
    https://www.sciencedirect.com/science/article/pii/S1361841521002516

    This datamodule use (A) for train, validation, test
    """

    def __init__(
        self,
        data_dir: str = "./",
        img_size: int = 256,
        num_workers: int = 0,
        batch_size: int = 16,
        pin_memory=False,
        drop_last=False,
        data_name="colon",
        data_ratio=1.0,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        resize_value = 256 if self.hparams.img_size == 224 else 456
        self.train_transform = Compose(
            [
                A.RandomResizedCrop(
                    height=self.hparams.img_size, width=self.hparams.img_size
                ),
                OneOf(
                    [
                        A.HorizontalFlip(p=1),
                        A.VerticalFlip(p=1),
                        A.RandomRotate90(p=1),
                    ]
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                OneOf(
                    [
                        A.GaussNoise(p=1),
                        A.GaussianBlur(p=1),
                        A.ColorJitter(),
                    ]
                ),
                A.RandomBrightness(limit=0.15, p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
        self.test_transform = Compose(
            [
                A.Resize(height=self.hparams.img_size, width=self.hparams.img_size),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

    @property
    def num_classes(self) -> int:
        return 4

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_df, valid_df = bring_colon_dataset_csv(
                datatype="COLON_MANUAL_512", stage=None
            )
            if self.hparams.data_ratio < 1.0:
                train_df = (
                    train_df.groupby("class")
                    .apply(lambda x: x.sample(frac=self.hparams.data_ratio))
                    .reset_index(drop=True)
                )
            self.train_dataset = ColonDataset(train_df, self.train_transform)
            self.valid_dataset = ColonDataset(valid_df, self.test_transform)
        else:
            test_df = bring_colon_dataset_csv(datatype="COLON_MANUAL_512", stage="test")
            self.test_dataset = ColonDataset(test_df, self.test_transform)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
            shuffle=False,
        )
