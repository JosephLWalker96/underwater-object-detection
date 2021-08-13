import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class QRDatasets(Dataset):

    # setting up member variable
    def __init__(self, dir_to_image: str, df: pd.DataFrame, transforms=None):
        super().__init__()

        self.num_class = 2  # QR code + background
        self.image_dir = dir_to_image
        self.dataframe = df
        self.transforms = transforms

    # returning the size of the data set
    def __len__(self) -> int:
        return self.dataframe.shape[0]

    # getting the data given certain index
    def __getitem__(self, idx: int):
        records = self.dataframe[self.dataframe.index == idx]
        img_path = self.image_dir + '/' + str(records["img_name"].values[0])
        
#         print(img_path)

        img = np.array(cv2.imread(img_path)).astype(np.float32)
        # # converting to grayscale
        img = np.array(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)).astype(np.float32)
        img /= 255.0

        boxes = records[['x', 'y', 'w', 'h']].values 
        
#         boxes[:, 2] = (boxes[:, 0] + boxes[:, 2])
#         boxes[:, 3] = (boxes[:, 1] + boxes[:, 3])
        
        boxes[:, 2] = int((boxes[:, 0] + boxes[:, 2])*float(records['image width'].values[0]))
        boxes[:, 3] = int((boxes[:, 1] + boxes[:, 3])*float(records['image height'].values[0]))
        boxes[:, 0] = int(boxes[:, 0]*float(records['image width'].values[0]))
        boxes[:, 1] = int(boxes[:, 1]*float(records['image height'].values[0]))
        
#         print(boxes)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)

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

        return img, target, idx
