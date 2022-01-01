import numpy as np
import torch
from ensemble_boxes import *

'''
    WBF (Weighted Box Fusion) approach for ensemble
'''

def make_ensemble_predictions(images, device, models):
    images = list(image.to(device) for image in images)
    result = []
    for net in models:
        net.eval()
        outputs = net(images)
        result.append(outputs)
    return result


def run_wbf(predictions, image_index, image_size=512, iou_thr=0.5, skip_box_thr=0.5, weights=None):
    boxes = np.array([prediction[image_index]['boxes'].data.cpu().numpy() / image_size for prediction in predictions])
#     boxes = [prediction[image_index]['boxes'].data.cpu().numpy() for prediction in predictions]
    scores = np.array([prediction[image_index]['scores'].data.cpu().numpy() for prediction in predictions])
#     print(scores)
    labels = np.array([prediction[image_index]['labels'].data.cpu().numpy() for prediction in predictions])
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr,
                                                  skip_box_thr=skip_box_thr)
    boxes = boxes * (image_size - 1)
    return boxes, scores, labels
