import copy
import os.path

import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from RandAugment import RandAugment
from PIL import Image


class QRDatasets(Dataset):

    # setting up member variable
    def __init__(self, dir_to_dataset: str, df: pd.DataFrame, use_grayscale: bool = False,
                 transforms=None, isTrain:bool = True, augment_list:list = None):
        super().__init__()

        self.isTrain = isTrain
        self.num_class = 3  # SUIT + target + background
        self.dataset_dir = dir_to_dataset
        self.dataframe = df
        self.transforms = transforms
        self.gray_scale = use_grayscale
        self.augment_list = augment_list

        # image name series from df
        self.img_name_ls = self.dataframe['Image'].unique()

    # returning the size of the data set
    def __len__(self) -> int:
        return len(self.img_name_ls)

    # getting the data given certain index
    def __getitem__(self, idx: int):
        img_name = self.img_name_ls[idx]
        records = self.dataframe.loc[self.dataframe['Image'] == img_name]

        img = None
        for idx, record in records.iterrows():
            img_path = os.path.join(self.dataset_dir, str(record["img_path"]))
            if self.gray_scale:
                # converting to grayscale
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                # change it back to BGR format
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        assert img is not None
        boxes = records[['x', 'y', 'w', 'h']].values

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        # labels = torch.ones((records.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(records['Labels'].values)

        hasObject = True

        for label in labels:
            # label 0 means no object in the image
            if label == 0:
                hasObject = False

        # all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        if hasObject:
            target['boxes'] = boxes
            target['labels'] = labels
        else:
            target['boxes'] = torch.as_tensor([[0, 0, 0.1, 0.1]])
            target['labels'] = torch.as_tensor([0])

        target['image_id'] = torch.tensor([idx])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': img,
                # 'image': np.array(img, dtype=np.float32)/255.0,
                'bboxes': target['boxes'],
                'labels': labels,
                'augment_list': self.augment_list # this is for RandAug Transform
            }
            sample = self.transforms(**sample)
            img = sample['image']

            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        if not hasObject:
            target['boxes'] = torch.as_tensor([[0, 0, 0.1, 0.1]])
            target['labels'] = torch.as_tensor([0])

        ##TODO: Normalize

        img = img.to(torch.float32)
        img /= 255.0

        return img, target, idx
