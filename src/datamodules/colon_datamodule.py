import albumentations as A
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from src.utils import bring_dataset_csv
from cv2 import cv2
import random
# from torchsampler import ImbalancedDatasetSampler


class CustomDataset(Dataset):
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
        return image['image'], label


class ColonDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "./",
            img_size: int = 256,
            num_workers: int = 4,
            batch_size: int = 32,
            pin_memory=False

    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        if self.hparams.img_size == 224:
            resize_value = 256
        else:
            resize_value = 456

        # Train augmentation policy
        self.train_transform = Compose(
            [
                OneOf([
                        Compose([
                            A.Resize(height=resize_value, width=resize_value),
                            A.RandomCrop(self.hparams.img_size, self.hparams.img_size)
                        ]),

                        A.CenterCrop(self.hparams.img_size, self.hparams.img_size, p=1),

                        A.Resize(height=self.hparams.img_size, width=self.hparams.img_size),
                    ], p=1),

                # A.RandomResizedCrop(height=self.hparams.img_size, width=self.hparams.img_size),

                A.HorizontalFlip(p=0.6),
                # Flip the input horizontally around the y-axis.
                A.VerticalFlip(p=0.6),
                # Flip the input Vertically around the x-axis.
                A.RandomRotate90(p=0.6),

                # A.ShiftScaleRotate(
                #     shift_limit=0.05,
                #     scale_limit=0.05,
                #     rotate_limit=15,
                #     p=0.5
                # ),
                # Randomly apply affine transforms: translate, scale and rotate the input

                # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
                # color augmentation

                # OneOf([
                #
                #     A.GaussNoise(p=0.5),
                #     # A_T.Adv
                #     # A.GaussianBlur(p=0.9),
                #     # A.ColorJitter(),
                #
                # ]),
                # A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),
                # A.RandomBrightness(limit=0.15, p=0.5),
                # Randomly change brightness of the input image.

                # A.Normalize(),
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
            train_df, valid_df = bring_dataset_csv(datatype='COLON_MANUAL_512', stage=None)
            random.seed(42)

            # Train dataset
            self.train_dataset = CustomDataset(train_df, self.train_transform)
            # Validation dataset
            self.valid_dataset = CustomDataset(valid_df, self.test_transform)
            # Test dataset
        else:
            test_df = bring_dataset_csv(datatype='COLON_MANUAL_512', stage='test')
            random.seed(42)
            self.test_dataset = CustomDataset(test_df, self.test_transform)

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
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            shuffle=False,
        )
