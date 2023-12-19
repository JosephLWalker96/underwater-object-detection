import os
import xml.etree.ElementTree as ET
import random

import chainer
import numpy as np
from chainercv.datasets.voc import voc_utils
from chainercv.utils import read_image
from RandAugment import *
import cv2
import PIL.Image as Image

from opt import bam_contents_classes


class BaseDetectionDataset(chainer.dataset.DatasetMixin):
    def __init__(self, root, subset, use_difficult, return_difficult, is_train=False):
        self.root = root
        self.img_dir = os.path.join(root, 'JPEGImages')
        self.imgset_dir = os.path.join(root, 'ImageSets/Main')
        self.ann_dir = os.path.join(root, 'Annotations')
        id_list_file = os.path.join(
            self.imgset_dir, '{:s}.txt'.format(subset))
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.subset = subset
        self.labels = None  # for network
        self.actual_labels = None  # for visualization
        self.is_train = is_train

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.ann_dir, id_ + '.xml'))
        bbox = []
        label = []
        difficult = []
        objs = anno.findall('object')

        for obj in objs:
            # If not using difficult split, and the object is
            # difficult, skip it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            bndbox_anno = obj.find('bndbox')
            name = obj.find('name').text.lower().strip()
            if name == 'suit':
                name = 'SUIT'
            label.append(self.labels.index(name))
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            difficult.append(int(obj.find('difficult').text))

        try:
            bbox = np.stack(bbox).astype(np.float32)
            label = np.stack(label).astype(np.int32)
            difficult = np.array(difficult, dtype=np.bool)
        except ValueError:
            i = random.choice(np.arange(0, len(self.ids)))
            return self.get_example(i)

        # Load a image
        try:
            img_file = os.path.join(self.img_dir, id_ + '.JPG')
            if os.path.exists(img_file):
#                 img = read_image(img_file, color=True)
                img = cv2.imread(img_file)
            else:
                img_file = os.path.join(self.img_dir, id_ + '.png')
                # img = read_image(img_file, color=True)
                img = cv2.imread(img_file)
            assert img is not None
        except:
            i = random.choice(np.arange(0, len(self.ids)))
            return self.get_example(i)
        
#         print(cv2.imread(img_file).transpose((2, 0, 1)))
#         print(img)
        
        im = img.transpose((1, 0, 2))
        bboxes = bbox

        if self.is_train:
            # shape transform (changed shape+position)
            if np.random.random(size=1)[0] > 0.5:
                m = np.random.randint(low=0, high=30, size=1)[0]
                dist_or_perspective = RandAugment(1, m, \
                                                [
                                                    ('Perspective', 0, 0.5), \
                                                    ('FurtherDistance', 0.1, 1)
                                                ])
                crop = RandAugment(1, m,
                            [('RandomCropping', 0, 0.5)])
                shape_transform = random.choice([dist_or_perspective, crop])
                im, bboxes = shape_transform(Image.fromarray(im), bboxes)
                im = np.array(im)

            # position transform (shape should be unchanged)
            if np.random.random(size=1)[0] > 0.5:
                n, m = np.random.randint(low=1, high=2, size=1)[0], np.random.randint(low=0, high=30, size=1)[0]
                randAug = RandAugment(n, m, \
                                [
                                    ('TranslateBboxSafe', 0, 1), \
                                    ('Rotate', 0, 30)
                                ])
                im, bboxes = randAug(Image.fromarray(im), bboxes)
                im = np.array(im)

        img = im.transpose((2, 1, 0))
        bbox = bboxes
        
        try:
            assert (len(bbox)) > 0
        except:
            i = random.choice(np.arange(0, len(self.ids)))
            return self.get_example(i)

        if self.return_difficult:
            return img, bbox, label, difficult
        return img, bbox, label


class VOCDataset(BaseDetectionDataset):
    def __init__(self, root, subset, use_difficult=False,
                 return_difficult=False):
        super(VOCDataset, self).__init__(root, subset, use_difficult,
                                         return_difficult)
        self.labels = voc_utils.voc_bbox_label_names
        self.actual_labels = voc_utils.voc_bbox_label_names


class ClipArtDataset(BaseDetectionDataset):
    def __init__(self, root, subset, use_difficult=False,
                 return_difficult=False):
        super(ClipArtDataset, self).__init__(root, subset, use_difficult,
                                             return_difficult)
        self.labels = voc_utils.voc_bbox_label_names
        self.actual_labels = voc_utils.voc_bbox_label_names


class BAMDataset(BaseDetectionDataset):
    def __init__(self, root, subset, use_difficult=False,
                 return_difficult=False):
        super(BAMDataset, self).__init__(root, subset, use_difficult,
                                         return_difficult)
        self.labels = voc_utils.voc_bbox_label_names
        self.actual_labels = bam_contents_classes

        
class SUITDataset(BaseDetectionDataset):
    def __init__(self, root, subset, use_difficult=False,
                 return_difficult=False, is_train=False):
        super(SUITDataset, self).__init__(root, subset, use_difficult,
                                             return_difficult, is_train)
        self.labels = ('SUIT')
        self.actual_labels = ('SUIT')
        