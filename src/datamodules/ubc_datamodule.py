import albumentations as A
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from cv2 import cv2
from glob import glob
from sklearn.model_selection import train_test_split
# from torchsampler import ImbalancedDatasetSampler
from collections import Counter



class UbcDataset(Dataset):
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

class UbcDataset_pd(Dataset):
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
    
def splitting(dataset): # train val test 80/10/10
    train, rest = train_test_split(dataset, train_size=0.8, shuffle=False,random_state=42)
    valid, test = train_test_split(rest, test_size=0.5, shuffle=False,random_state=42)
    return train, valid, test

def make_dataset(stage='train'):
    
    base_path='/home/compu/jh/data/prostate_miccai_2019_patches_690_80_step05/'
    # base_path='/home/compu/jh/data/Harvard_ReplicationData_Prostate'
    files=glob(f'{base_path}/*/*.jpg')

    data_class0=[data for data in files if int(data.split('_')[-1].split('.')[0])==0]
    data_class2=[data for data in files if int(data.split('_')[-1].split('.')[0])==2]
    data_class3=[data for data in files if int(data.split('_')[-1].split('.')[0])==3]
    data_class4=[data for data in files if int(data.split('_')[-1].split('.')[0])==4]

    train_data0, validation_data0, test_data0=splitting(data_class0)
    train_data2, validation_data2, test_data2=splitting(data_class2)
    train_data3, validation_data3, test_data3=splitting(data_class3)
    train_data4, validation_data4, test_data4=splitting(data_class4)
    label_dict = {0: 0, 2: 1, 3: 2, 4: 3}
    if stage=='train':
        
        train_path=train_data0+train_data2+train_data3+train_data4
        valid_path=validation_data0+validation_data2+validation_data3+validation_data4

        train_label=[int(path.split('.')[0][-1]) for path in train_path]
        valid_label=[int(path.split('.')[0][-1]) for path in valid_path]
        
        train_label = [label_dict[k] for k in train_label]
        valid_label = [label_dict[k] for k in valid_label]

        train_set=list(zip(train_path, train_label))
        valid_set=list(zip(valid_path, valid_label))
        return train_set, valid_set
    else:
        test_path=test_data0+test_data2+test_data3+test_data4        
        test_label=[int(path.split('.')[0][-1]) for path in test_path]
        test_label = [label_dict[k] for k in test_label]

        test_set=list(zip(test_path, test_label))
    return test_set
# prostate_miccai_2019_patches_690_80_step05
# class 0: 1811
# class 2: 7037
# class 3: 11431
# class 4: 292
# 1284 BN, 5852 grade 3, 9682 grade 4, and 248 grade 5

# Havard_ReplicationData_Prostate
# class 0: 2869
# class 1: 8828
# class 2: 7235
# class 3: 3090


class UbcDataModule(LightningDataModule):
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

            train_set, valid_set = make_dataset(stage="train")
            self.train_dataset = UbcDataset(train_set, self.train_transform)
            self.valid_dataset = UbcDataset(valid_set, self.test_transform)

        else:
            test_set = make_dataset(stage="test")
            self.test_dataset = UbcDataset(test_set, self.test_transform)

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
