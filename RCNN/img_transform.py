import copy

import albumentations as A
import numpy as np
from PIL import Image
from albumentations.pytorch import ToTensorV2

from color_correction import color_correction
from typing import Union, Sequence, Optional, Tuple, List, Dict, Any

import sys
sys.path.append('../')
from RandAugment import RandAugment

def get_train_val_transform(type: str = "color_correction"):
    return generate_transform(type)


def get_test_transform(type: str = "no_transform"):
    return generate_transform(type)


class Color_Correction(A.DualTransform):
    def apply(self, img, **params) -> np.ndarray:
        mu, sigma = 0, .3  # mean and standard deviation
        s1, s2 = np.random.normal(mu, sigma, 2)
        img = color_correction(pixels=img, x=s1, y=s2, adjustment_intensity=1)
        return img

    def apply_to_bbox(self, bbox, **params):
        return bbox


def RandAug(image, bboxes, labels, augment_list):
    img = np.array(image)
    boxes = np.array(bboxes)
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

    p = np.random.random(size=1)[0]
    if p > 0.7:
        while True:
            need_rerun = False
            n, m = np.random.randint(low=1, high=4, size=1)[0], np.random.randint(low=0, high=30, size=1)[0]
            randAug = RandAugment(n, m, augment_list)
            t_img, t_boxes = randAug(Image.fromarray(copy.deepcopy(img)), copy.deepcopy(boxes))
            t_img = np.array(t_img)

            # checking whether the augmentation is valid
            t_boxes[:, 2] = t_boxes[:, 0] + t_boxes[:, 2]
            t_boxes[:, 3] = t_boxes[:, 1] + t_boxes[:, 3]
            t_boxes = np.where(t_boxes > 0, t_boxes, 0)
            t_boxes[:, 0] = np.where(t_boxes[:, 0] < img.shape[1], t_boxes[:, 0], img.shape[1])
            t_boxes[:, 1] = np.where(t_boxes[:, 1] < img.shape[0], t_boxes[:, 1], img.shape[0])
            t_boxes[:, 2] = np.where(t_boxes[:, 2] < img.shape[1], t_boxes[:, 2], img.shape[1])
            t_boxes[:, 3] = np.where(t_boxes[:, 3] < img.shape[0], t_boxes[:, 3], img.shape[0])

            for t_box in t_boxes:
                if (t_box[0] >= t_box[2]) or (t_box[1] >= t_box[3]):
                    need_rerun = True

            if not need_rerun:
                boxes = t_boxes
                img = t_img
                break
    else:
        boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] + boxes[:, 1]

    transform = A.Compose([
        A.Resize(512, 512),
        # Color_Correction(),
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    sample = {
        'image': img,
        # 'image': np.array(img, dtype=np.float32)/255.0,
        'bboxes': boxes,
        'labels': labels
    }
    return transform(**sample)


# there are three type of transform:
#   "default": mild augmentation with low p value,
#   "color_correction": only do color_correction,
#   "intensive" : augmentation + color_correction with high p value,
#   else no transform, simply resize and transfer to tensor
def generate_transform(type="default"):
    if type == "default":
        return A.Compose([
            A.Resize(512, 512),
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(p=0.0),
            A.RandomGamma(p=0.0),
            A.ImageCompression(quality_lower=75, p=0.0),
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    elif type == "color_correction":
        return A.Compose([
            A.Resize(512, 512),
            Color_Correction(),
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    elif type == 'intensive':
        return A.Compose([
            A.Resize(512, 512),
            Color_Correction(),
            A.CLAHE(p=0.5),
            A.RandomGamma(p=0.5),
            # A.OpticalDistortion(p=0.5),
            A.RandomContrast(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightness(p=0.5),
            # A.RandomCropNearBBox(p=0.5),
            A.RandomToneCurve(p=0.5),
            A.ImageCompression(quality_lower=75, p=0.1),
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    elif type == 'RandAug':
        return RandAug
    else:
        return A.Compose([
            A.Resize(512, 512),
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
