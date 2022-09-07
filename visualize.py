import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import sklearn.cluster
import torch

path1='/home/compu/jh/data/voting/features'
path2='/home/compu/jh/data/voting/targets'
path3='/home/compu/jh/data/voting/max_probs'
path4='/home/compu/jh/data/voting/preds'


name='ubc_densenet121_0.5.npy'
features=np.load(f'{path1}/{name}')
targets=np.load(f'{path2}/{name}')
max_probs=np.load(f'{path3}/{name}')
preds=np.load(f'{path4}/{name}')

# feature_by_cls=[features[np.logical_and(targets==i,preds==i)] for i in range(4)]
random_idxs = [
                np.random.choice(
                    np.where((preds == i)&(targets == i))[0],
                    10,
                    replace=False,
                )
                for i in range(4)
            ]
print()
# from src.utils import bring_dataset_csv

# train_df, valid_df = bring_dataset_csv(datatype='COLON_MANUAL_512', stage=None)
# print(train_df)
# train_df.groupby('class').apply(lambda x:x.sample(frac=0.5,random_state=42)).reset_index(drop=True)


# random_idxs = [
#     np.random.choice(
#         np.where(np.logical_and(preds == i, max_probs > 0.9, targets==i))[0],
#         10,
#         replace=False,
#     )
#     for i in range(4)
# ]

