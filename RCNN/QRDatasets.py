import copy
import os.path

import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from RandAugment import RandAugment
from PIL import Image
import time


class QRDatasets(Dataset):

    # setting up member variable
    def __init__(self, dir_to_dataset: str, df: pd.DataFrame, use_grayscale: bool = False,
                 transforms=None, isTrain: bool = True, augment_list: list = None, test_df: pd.DataFrame = None):
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

        # taking example from dataset to use color matcher as transform function
        self.test_examples = None
        self.CM = None
        if test_df is not None:
            from color_matcher import ColorMatcher
            self.CM = ColorMatcher()
            self.test_examples = test_df["img_path"].unique()


    def __prepare_cm__(self):
        st = time.time()
        if self.test_examples is not None and self.CM is not None:
            test_img_paths = np.random.choice(self.test_examples, 10)
            self.img_refs = []
            for test_img_path in test_img_paths:
                from color_matcher.io_handler import load_img_file
                test_img_path = os.path.join(self.dataset_dir, str(test_img_path))
                img_ref = load_img_file(test_img_path)
                self.img_refs.append(img_ref)
#         print('preloading ref img takes %f'%(time.time()-st))

    # returning the size of the data set
    def __len__(self) -> int:
        return len(self.img_name_ls)

    # getting the data given certain index
    def __getitem__(self, idx: int):
        img_name = self.img_name_ls[idx]
        records = self.dataframe.loc[self.dataframe['Image'] == img_name]

        img = None
        domain_label = 0 # 0 for source domain (without color-matcher), otherwise 1
        for idx, record in records.iterrows():
            img_path = os.path.join(self.dataset_dir, str(record["img_path"]))
            if self.gray_scale:
                # converting to grayscale
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                # change it back to BGR format
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            if self.test_examples is not None and self.CM is not None and np.random.random(size=1)[0] > 0.5:
                try:
                    from color_matcher.io_handler import load_img_file
                    from color_matcher.normalizer import Normalizer
                    # test_img_path = np.random.choice(self.test_examples, 1)[0]
                    # test_img_path = os.path.join(self.dataset_dir, str(test_img_path))
                    # img_ref = load_img_file(test_img_path)
                    img_ref_idx = np.random.choice(np.arange(5), 1)[0]
                    img_ref = self.img_refs[img_ref_idx]
                    img_src = load_img_file(img_path)
                    st = time.time()
                    img = self.CM.transfer(src=img_src, ref=img_ref, method='hm-mvgd-hm')
                    # METHODS = ['default', 'hm', 'reinhard', 'mvgd', 'mkl', 'hm-mvgd-hm', 'hm-mkl-hm']
                    # img = self.CM.transfer(src=img_src, ref=img_ref, method=np.random.choice(METHODS, 1)[0])
#                     print('transferring img takes %f'%(time.time()-st))
                    # normalize image intensity to 8-bit unsigned integer
                    st = time.time()
                    img = Normalizer(img).uint8_norm()
#                     print('normalizing img takes %f'%(time.time()-st))
                    domain_label = 1
                except:
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
                'augment_list': self.augment_list  # this is for RandAug Transform
            }
            sample = self.transforms(**sample)
            img = sample['image']

            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        if not hasObject:
            target['boxes'] = torch.as_tensor([[0, 0, 0.1, 0.1]])
            target['labels'] = torch.as_tensor([0])

        img = img.to(torch.float32)
        img /= 255.0

        return img, target, idx, domain_label
