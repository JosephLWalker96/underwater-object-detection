from abc import ABC

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

from color_correction import color_correction


def get_train_val_transform(type:str = "color_correction"):
    return generate_transform()


def get_test_transform(type:str = "no_transform"):
    return generate_transform()


class Color_Correction(A.ImageOnlyTransform):
    def apply(self, img, **params) -> np.ndarray:
        mu, sigma = 0, .3  # mean and standard deviation
        s1, s2 = np.random.normal(mu, sigma, 2)
        img = color_correction(pixels=img, x=s1, y=s2, adjustment_intensity=1)
        return img


# there are three type of transform:
#   "default": mild augmentation with low p value,
#   "color_correction": only do color_correction,
#   "intensive" : augmentation + color_correction with high p value,
#   else no transform, simply resize and transfer to tensor
def generate_transform(type="default"):
    if type == "default":
        return A.Compose([
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
            Color_Correction(),
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    elif type == 'intensive':
        return A.Compose([
            A.CLAHE(p=0.5),
            A.RandomGamma(p=0.5),
            # A.OpticalDistortion(p=0.5),
            A.RandomContrast(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightness(p=0.5),
            # A.RandomCropNearBBox(p=0.5),
            Color_Correction(),
            A.RandomToneCurve(p=0.5),
            A.ImageCompression(quality_lower=75, p=0.1)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(512, 512),
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
