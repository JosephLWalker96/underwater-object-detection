import numpy as np
import pandas as pd
import torch


class ScoreMeasurer:
    def __init__(self):
        # elements for mAP
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0

    def get_precision(self):
        if self.true_positive == 0:
            return 0
        return self.true_positive / (self.true_positive + self.false_positive)

    def get_recall(self):
        if self.true_positive == 0:
            return 0
        return self.true_positive / (self.true_positive + self.false_negative)

    def get_F_Measure(self):
        if self.true_positive == 0:
            return 0
        return 2 / (1 / self.get_recall() + 1 / self.get_precision())


'''
    special case:
        iou == -1: true negative (in this case, the args 'output' is an empty set)
        iou == -2: misclassified target as SUIT or SUIT as target
        iou == -3: false positive 
'''


def _get_iou_score(output, target, width, height, target_idx):
    output_boxes = output['boxes']
    output_scores = output['scores']
    output_labels = output['labels']

    best_iou = 0
    best_box = None

    target_area = target['area'][target_idx].item()
    target_box = target['boxes'][target_idx]
    target_label = target['labels'][target_idx]

    # label==0 means no label
    if target_label == 0:
        if len(output_boxes) == 0:
            return None, -1
        else:
            return output_boxes, -3

    target_x1 = target_box[0].item() / width
    target_x2 = target_box[2].item() / width
    target_y1 = target_box[1].item() / height
    target_y2 = target_box[3].item() / height
    target_area = (target_y2 - target_y1) * (target_x2 - target_x1)

    for i in range(len(output_boxes)):
        box = output_boxes[i]
        score = output_scores[i]
        label = output_labels[i]
        if score < 0.5:
            continue

        box_x1 = box[0].item()
        box_y1 = box[1].item()
        box_x2 = box[2].item()
        box_y2 = box[3].item()
        output_area = (box_y2 - box_y1) * (box_x2 - box_x1)

        merge_x1 = max([target_x1, box_x1])
        merge_x2 = min([target_x2, box_x2])
        merge_y1 = max([target_y1, box_y1])
        merge_y2 = min([target_y2, box_y2])

        merge_area = 0
        iou = 0.0
        if merge_y2 - merge_y1 > 0 and merge_x2 - merge_x1 > 0:
            merge_area = (merge_y2 - merge_y1) * (merge_x2 - merge_x1)
            iou = merge_area / (output_area + target_area - merge_area)

        if output.__contains__('IoU'):
            if label != target_label or iou == 0:
                output['IoU'][i] = max(output['IoU'][i], -2)
            else:
                output['IoU'][i] = max(output['IoU'][i], iou)

        if iou > best_iou:
            best_iou = iou
            best_box = box

    return best_box, best_iou


def get_iou_score(output, target, width, height):
    cpu = torch.device('cpu')
    if type(output['boxes']) is torch.Tensor:
        output_boxes = output['boxes'].to(cpu).clone().detach()
        output_boxes.requires_grad = False
    else:
        output_boxes = torch.tensor(output['boxes'], device=cpu, requires_grad=False)
    if type(output['labels']) is torch.Tensor:
        output_labels = output['labels'].to(cpu).clone().detach()
        output_labels.requires_grad = False
    else:
        output_labels = torch.tensor(output['labels'], device=cpu, requires_grad=False)
    n = len(output_labels)

    target_boxes = target['boxes'].to(cpu)
    target_boxes[:, 0] /= width
    target_boxes[:, 1] /= height
    target_boxes[:, 2] /= width
    target_boxes[:, 3] /= height

    target_labels = target['labels'].to(cpu)
    m = len(target_labels)

    # it should have a shape of (len(output), len(target)) = (n, m)
    label_matching_matrix = torch.where(output_labels.unsqueeze(dim=1).expand(n, m) == target_labels.unsqueeze(dim=0),
                                        1, -1)

    # label==0 means no label
    for target_label in target_labels:
        if target_label == 0:
            if len(output_boxes) == 0:
                return -1
            else:
                return -3

    # getting union area
    target_area = (target_boxes[:, 3] - target_boxes[:, 1]) * (target_boxes[:, 2] - target_boxes[:, 0])
    output_area = (output_boxes[:, 3] - output_boxes[:, 1]) * (output_boxes[:, 2] - output_boxes[:, 0])
    union_area = output_area.unsqueeze(dim=1).expand(n, m) + target_area.unsqueeze(dim=0).expand(n, m)

    # getting merged bbox region
    merged_x0y0 = torch.max(output_boxes[:, :2].unsqueeze(dim=1).expand(n, m, 2),
                            target_boxes[:, :2].unsqueeze(dim=0).expand(n, m, 2))
    merged_x1y1 = torch.min(output_boxes[:, 2:].unsqueeze(dim=1).expand(n, m, 2),
                            target_boxes[:, 2:].unsqueeze(dim=0).expand(n, m, 2))
    merged_wh = torch.clamp((merged_x1y1 - merged_x0y0), min=0)
    merged_area = merged_wh[:, :, 0] * merged_wh[:, :, 1]

    iou_matrix = torch.multiply(merged_area / (union_area-merged_area), label_matching_matrix)
    iou_matrix = torch.where(iou_matrix >= 0, abs(iou_matrix.to(float)), float(-2))
    iou = torch.amax(iou_matrix, dim=1)
    output['IoU'] = iou
    output['matched_target'] = torch.argmax(iou_matrix, dim=1)
    output['num_target'] = m

    if len(iou) > 0:
        return torch.mean(torch.where(iou >= 0, iou.to(float), float(0)))
    else:
        return 0
