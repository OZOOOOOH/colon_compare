import albumentations as A
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from cv2 import cv2
from glob import glob
# from torchsampler import ImbalancedDatasetSampler
from collections import Counter

class HarvardDataset(Dataset):
    def __init__(self, pair_list, transform=None):
        self.pair_list = pair_list
        self.transform = transform

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        pair = self.pair_list[idx]
        image = cv2.imread(pair[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = pair[1]
        if self.transform:
            image = self.transform(image=image)

        return image["image"], label

class HarvardDataset_pd(Dataset):
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


class ProstateDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        img_size: int = 256,
        num_workers: int = 4,
        batch_size: int = 32,
        pin_memory=False,
        drop_last=True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        resize_value = 256 if self.hparams.img_size == 224 else 456
        # Train augmentation policy
        self.train_transform = Compose(
            [
                A.RandomResizedCrop(height=self.hparams.img_size, width=self.hparams.img_size),
                OneOf(
                    [
                        A.HorizontalFlip(p=1),
                        # Flip the input horizontally around the y-axis.
                        A.VerticalFlip(p=1),
                        # Flip the input Vertically around the x-axis.
                        A.RandomRotate90(p=1),
                    ]
                ),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                # Randomly apply affine transforms: translate, scale and rotate the input
                # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
                # color augmentation
                OneOf(
                    [
                        # A.MotionBlur(p=1),
                        # A.OpticalDistortion(p=1),
                        A.GaussNoise(p=1),
                        # A_T.Adv
                        A.GaussianBlur(p=1),
                        A.ColorJitter(),
                    ]
                ),
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

            train_set, valid_set = prepare_prostate_harvard_data(stage="train")
            self.train_dataset = HarvardDataset(train_set, self.train_transform)
            self.valid_dataset = HarvardDataset(valid_set, self.test_transform)

        else:
            test_set = prepare_prostate_harvard_data(stage="test")
            self.test_dataset = HarvardDataset(test_set, self.test_transform)

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
