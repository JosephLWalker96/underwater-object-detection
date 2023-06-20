# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import cfg
from roi_data_layer.minibatch import get_minibatch
from utils.blob import prep_im_for_blob, im_list_to_blob
from RandAugment import *
import cv2
import numpy as np
import numpy.random as npr
import random
import PIL.Image as Image
import time
import os
from torch.utils.data import DataLoader, Dataset

class RoIDataLayer(object):
  """Fast R-CNN data layer used for training."""

  def __init__(self, roidb, num_classes, random=False):
    """Set the roidb to be used by this layer during training."""
    self._roidb = roidb
    self._num_classes = num_classes
    # Also set a random flag
    self._random = random
    self._shuffle_roidb_inds()
    
    self.dataloader = DataLoader(MyDataset(roidb), 
                                 shuffle=random, 
                                 batch_size=1,
                                 pin_memory=True, 
                                 num_workers=os.cpu_count())
    self.iter = self.dataloader.__iter__()

  def _shuffle_roidb_inds(self):
    """Randomly permute the training roidb."""
    # If the random flag is set, 
    # then the database is shuffled according to system time
    # Useful for the validation set
    if self._random:
      st0 = np.random.get_state()
      millis = int(round(time.time() * 1000)) % 4294967295
      #np.random.seed(millis)
    
    if cfg.TRAIN.ASPECT_GROUPING:
      widths = np.array([r['width'] for r in self._roidb])
      heights = np.array([r['height'] for r in self._roidb])
      horz = (widths >= heights)
      vert = np.logical_not(horz)
      horz_inds = np.where(horz)[0]
      vert_inds = np.where(vert)[0]
      inds = np.hstack((
          np.random.permutation(horz_inds),
          np.random.permutation(vert_inds)))
      inds = np.reshape(inds, (-1, 2))
      row_perm = np.random.permutation(np.arange(inds.shape[0]))
      inds = np.reshape(inds[row_perm, :], (-1,))
      self._perm = inds
    else:
      self._perm = np.random.permutation(np.arange(len(self._roidb)))
    ##no shuffle
    self._perm = np.arange(len(self._roidb))
    # Restore the random state
    #if self._random:
      #np.random.set_state(st0)
      
    self._cur = 0

  def _get_next_minibatch_inds(self):
    """Return the roidb indices for the next minibatch."""
    
    if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
      self._shuffle_roidb_inds()

    db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
    self._cur += cfg.TRAIN.IMS_PER_BATCH

    return db_inds

  def _get_next_minibatch(self):
    """Return the blobs to be used for the next minibatch.

    If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
    separate process and made available through self._blob_queue.
    """
    db_inds = self._get_next_minibatch_inds()
    minibatch_db = [self._roidb[i] for i in db_inds]
    return get_minibatch(minibatch_db, self._num_classes)
      
  def forward(self):
    """Get blobs and copy them into this layer's top blob vector."""
    # blobs = self._get_next_minibatch()
    try:
      blobs = self.iter.__next__()
    except StopIteration:
      self.iter = self.dataloader.__iter__()
      blobs = self.iter.__next__()

    blobs['data'] = blobs['data'].squeeze(dim=0).cpu().detach().numpy()
    blobs['gt_boxes'] = blobs['gt_boxes'].squeeze(dim=0).cpu().detach().numpy()
    blobs['im_info'] = blobs['im_info'].squeeze().cpu().detach().numpy()
    
    # print(blobs['data'].shape)
    # print(blobs['gt_boxes'].shape)
    # print(blobs['im_info'].shape)
    return blobs

class MyDataset(Dataset):
  def __init__(self, roidb):
    super().__init__()
    self.database = roidb

  def __len__(self) -> int:
    return len(self.database)

  def __getitem__(self, idx: int):
    data_entry = self.database[idx]
    scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=1)

    i = 0
    processed_ims = []
    im_scales = []
    im_path = []
    im = cv2.imread(data_entry['image'])
    orig_imshape = im.shape
    if data_entry['flipped']:
      im = im[:, ::-1, :]

    # gt boxes: (x1, y1, x2, y2, cls)
    if cfg.TRAIN.USE_ALL_GT:
      # Include all ground truth boxes
      gt_inds = np.where(data_entry['gt_classes'] != 0)[0]
    else:
      # For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
      gt_inds = np.where(data_entry['gt_classes'] != 0 & np.all(data_entry['gt_overlaps'].toarray() > -1.0, axis=1))[0]
    
    bboxes = data_entry['boxes'][gt_inds, :]

#     shape transform (changed shape+position)
#     if np.random.random(size=1)[0] > 0.5:
#         m = np.random.randint(low=0, high=30, size=1)[0]
#         dist_or_perspective = RandAugment(1, m, \
#                                           [('FurtherDistance', 0.1, 1)])
# #                                           [('Perspective', 0, 0.5), \
# #                                            ('FurtherDistance', 0.1, 1)])
#         crop = RandAugment(1, m,
#                           [('RandomCropping', 0, 0.5)])
#         shape_transform = random.choice([dist_or_perspective, crop])
#         im, bboxes = shape_transform(Image.fromarray(im), bboxes)
#         im = np.array(im)
    # position transform (shape should be unchanged)
#     if np.random.random(size=1)[0] > 0.5:
#         n, m = np.random.randint(low=1, high=2, size=1)[0], np.random.randint(low=0, high=30, size=1)[0]
#         randAug = RandAugment(n, m, \
#                               [('TranslateBboxSafe', 0, 1), \
#                               ('Rotate', 0, 30)])
#         im, bboxes = randAug(Image.fromarray(im), bboxes)
#         im = np.array(im)
    
#     m = np.random.randint(low=0, high=30, size=1)[0]
#     randAug =  RandAugment(1, m, [('RandomCropping', 0, 0.5)])
#     randAug =  RandAugment(1, m, [('FurtherDistance', 0.1, 1)])
#     randAug =  RandAugment(1, m, [('Perspective', 0, 0.5)])
    n, m = np.random.randint(low=1, high=2, size=1)[0], np.random.randint(low=0, high=30, size=1)[0]
#     randAug = RandAugment(n, m, [('TranslateBboxSafe', 0, 1)])
    randAug = RandAugment(n, m, [('Rotate', 0, 30)])
#                               , \
#                               ('Rotate', 0, 30)])
    
    shape_transform = randAug
    im, bboxes = shape_transform(Image.fromarray(im), bboxes)
    im = np.array(im)

    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = bboxes
    gt_boxes[:, 4] = data_entry['gt_classes'][gt_inds]

    
    im_path.append(data_entry['image'])
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)
    gt_boxes[:, 0:4] *= im_scale

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    blobs = {'data': blob}
    blobs['data_path'] = im_path
    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array(
      [blob.shape[1], blob.shape[2], im_scales[0], orig_imshape[0], orig_imshape[1], orig_imshape[2]],
      dtype=np.float32)  

    # print(blobs['data'].shape)
    # print(blobs['gt_boxes'].shape)
    # print(blobs['im_info'].shape)

    return blobs

      