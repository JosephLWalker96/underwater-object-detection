# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
from RandAugment import *
import PIL.Image as Image
import random

def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)

  # Sample random scales to use for each image in this batch
  scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)

  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  assert len(roidb) == 1, "Single batch only"

  i = 0
  processed_ims = []
  im_scales = []
  im_path = []
  im = cv2.imread(roidb[i]['image'])
  orig_imshape = im.shape
  if roidb[i]['flipped']:
    im = im[:, ::-1, :]

  # gt boxes: (x1, y1, x2, y2, cls)
  if cfg.TRAIN.USE_ALL_GT:
    # Include all ground truth boxes
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
  else:
    # For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
    gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
  
  bboxes = roidb[0]['boxes'][gt_inds, :]

#  if np.random.random(size=1)[0] > 0.5:
#      n, m = np.random.randint(low=1, high=4, size=1)[0], np.random.randint(low=0, high=30, size=1)[0]
#      randAug = RandAugment(1, m)
#      im, bboxes = randAug(Image.fromarray(im), bboxes)
#      im = np.array(im)
      
  # shape transform (changed shape+position)
  if np.random.random(size=1)[0] > 0.5:
       m = np.random.randint(low=0, high=30, size=1)[0]
       dist_or_perspective = RandAugment(1, m, \
                                         [('Perspective', 0, 0.5), \
                                          ('FurtherDistance', 0.1, 1)])
       crop = RandAugment(1, m,
                        [('RandomCropping', 0, 0.5)])
       shape_transform = random.choice([dist_or_perspective, crop])
       im, bboxes = shape_transform(Image.fromarray(im), bboxes)
#      im, bboxes = crop(Image.fromarray(im), bboxes)
       im = np.array(im)
  # position transform (shape should be unchanged)
  if np.random.random(size=1)[0] > 0.5:
      n, m = np.random.randint(low=1, high=2, size=1)[0], np.random.randint(low=0, high=30, size=1)[0]
      randAug = RandAugment(n, m, \
                            [('TranslateBboxSafe', 0, 1), \
                             ('Rotate', 0, 30)])
      im, bboxes = randAug(Image.fromarray(im), bboxes)
      im = np.array(im)

  gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes[:, 0:4] = bboxes
  gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]

  
  im_path.append(roidb[i]['image'])
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

  return blobs
