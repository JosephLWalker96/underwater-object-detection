import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class QRDatasets(Dataset):

    # setting up member variable
    def __init__(self, dir_to_image: str, df: pd.DataFrame, use_grayscale: bool = False,transforms=None):
        super().__init__()

        self.num_class = 2  # QR code + background
        self.image_dir = dir_to_image
        self.dataframe = df
        self.transforms = transforms
        self.gray_scale = use_grayscale

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
            img_path = self.image_dir + '/' + str(record["File Name"])

    #         print(img_path)
            if self.gray_scale:
                # converting to grayscale
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                # change it back to BGR format
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.imread(img_path)

        assert img is not None
    #         print(img.shape)

        boxes = records[['x', 'y', 'w', 'h']].values

#         boxes[:, 2] = (boxes[:, 0] + boxes[:, 2])
#         boxes[:, 3] = (boxes[:, 1] + boxes[:, 3])

        boxes[:, 2] = (boxes[:, 0] + boxes[:, 2])
        boxes[:, 3] = (boxes[:, 1] + boxes[:, 3])
        boxes[:, 0] = boxes[:, 0]
        boxes[:, 1] = boxes[:, 1]
#         print(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        # labels = torch.ones((records.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(records['Labels'].values)

        for label in labels:
            if label == 2:
                labels = torch.as_tensor([])
                boxes = torch.as_tensor([])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        target['area'] = area
        target['iscrowd'] = iscrowd

        
        if self.transforms:
            sample = {
                'image': img,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            img = sample['image']
            
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
        
#         img = img
#         img /= 255.0

        return img, target, idx
