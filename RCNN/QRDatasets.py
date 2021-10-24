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

    # returning the size of the data set
    def __len__(self) -> int:
        return self.dataframe.shape[0]

    # getting the data given certain index
    def __getitem__(self, idx: int):
        records = self.dataframe[self.dataframe.index == idx]
        img_path = self.image_dir + '/' + str(records["File Name"].values[0])
        
#         print(img_path) 

        img = None
        if self.gray_scale:
            # converting to grayscale
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
#             img = gray
#             img = img.reshape(img.shape[0], img.shape[1], 1)
            
            # change it back to BGR format
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            
#             img = np.array(img).astype(np.float32)
#             print(img.shape)
        else:
            img = cv2.imread(img_path)
            
        assert img is not None
        
#         print(img.shape)

        boxes = records[['x', 'y', 'w', 'h']].values 
        
#         boxes[:, 2] = (boxes[:, 0] + boxes[:, 2])
#         boxes[:, 3] = (boxes[:, 1] + boxes[:, 3])
        
        boxes[:, 2] = int((boxes[:, 0] + boxes[:, 2]))
        boxes[:, 3] = int((boxes[:, 1] + boxes[:, 3]))
        boxes[:, 0] = int(boxes[:, 0])
        boxes[:, 1] = int(boxes[:, 1])
        
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
        
#         img = img
#         img /= 255.0

        return img, target, idx
