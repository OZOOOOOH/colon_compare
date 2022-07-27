import albumentations as A
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from src.utils import bring_gastirc_dataset_csv
from cv2 import cv2
import random
import torch
import glob
# from torchsampler import ImbalancedDatasetSampler


class GastricDataset(Dataset):
    def __init__(self, df, transform=None):
        self.image_id = df["path"].values
        self.labels = df["label"].values
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
        return image['image'], label


class GastricDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "./",
            img_size: int = 256,
            num_workers: int = 4,
            batch_size: int = 32,
            pin_memory=False,
            drop_last=True

    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        resize_value = 256 if self.hparams.img_size == 224 else 456
        # Train augmentation policy
        self.train_transform = Compose(
            [
                A.RandomResizedCrop(height=self.hparams.img_size, width=self.hparams.img_size),

                OneOf([
                    A.HorizontalFlip(p=1),
                    # Flip the input horizontally around the y-axis.
                    A.VerticalFlip(p=1),
                    # Flip the input Vertically around the x-axis.
                    A.RandomRotate90(p=1),
                ]),

                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=15,
                    p=0.5
                ),
                # Randomly apply affine transforms: translate, scale and rotate the input

                # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
                # color augmentation

                OneOf([
                    # A.MotionBlur(p=1),
                    # A.OpticalDistortion(p=1),
                    A.GaussNoise(p=1),
                    # A_T.Adv
                    A.GaussianBlur(p=1),
                    A.ColorJitter(),

                ]),
                # A.cutout(),
                A.RandomBrightness(limit=0.15, p=0.5),
                # Randomly change brightness of the input image.
                A.Normalize(),
                # Normalization is applied by the formula: img = (img - mean * max_pixel_value) / (std * max_pixel_value)

                ToTensorV2(),
                # Convert image and mask to torch.Tensor

            ]
        )
        # Validation/Test augmentation policy
        self.test_transform = Compose(
            [
                A.Resize(height=self.hparams.img_size, width=self.hparams.img_size),
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.Normalize(),
                ToTensorV2(),

            ]

        )

    @property
    def num_classes(self) -> int:
        return 4

    def setup(self, stage=None):
        # Assign train/val/test datasets for use in dataloaders
        if stage == "fit" or stage is None:
            # Random train-validation split
            train_df, valid_df = bring_gastirc_dataset_csv(stage=None)
            # Train dataset
            self.train_dataset = GastricDataset(train_df, self.train_transform)
            # Validation dataset
            self.valid_dataset = GastricDataset(valid_df, self.test_transform)
            # Test dataset
        else:
            test_df = bring_gastirc_dataset_csv(stage='test')
            self.test_dataset = GastricDataset(test_df, self.test_transform)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            shuffle=True,
            # sampler=ImbalancedDatasetSampler(self.train_dataset),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
            shuffle=False,
        )
