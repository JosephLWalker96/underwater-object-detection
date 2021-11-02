import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class QRDatasets(Dataset):

    # setting up member variable
    def __init__(self, dir_to_image: str, df: pd.DataFrame, use_grayscale: bool = False,transforms=None):
        super().__init__()

        self.num_class = 3  # SUIT + target + background
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

            if self.gray_scale:
                # converting to grayscale
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                # change it back to BGR format
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.imread(img_path)

        assert img is not None

        boxes = records[['x', 'y', 'w', 'h']].values

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        boxes[:, 0] = boxes[:, 0]
        boxes[:, 1] = boxes[:, 1]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         for box in boxes:
#             width = float(records['Image Width'].values[0])
#             height = float(records['Image Height'].values[0])
#             if box[3]<=box[1]:
#                 print(img_name)
#             if box[2]<=box[0]:
#                 print(img_name)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        # labels = torch.ones((records.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(records['Labels'].values)

        for label in labels:
            if label == 2:
                boxes = torch.as_tensor([[-2, -2, -1, -1]])

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
#                 'image': img,
                'image': np.array(img, dtype=np.float32)/255.0,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            img = sample['image']
#             print(img)
            
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
        
#         img = img
#         img /= 255.0

        return img, target, idx
