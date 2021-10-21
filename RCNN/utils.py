import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def collate_fn(batch):
    return tuple(zip(*batch))


# Augmentations
def get_train_transform():
    return A.Compose([
        A.Resize(512, 512),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_test_transform():
    return A.Compose([
        A.Resize(512, 512),
        ToTensorV2(p=1.0)
    ])

def get_iou_score(output, target, width, height):

    output_boxes = output['boxes']
    output_scores = output['scores']
    
    best_iou = 0
    best_box = None
    
    target_area = target['area'].item()
    target_box = target['boxes'][0]
    # flag is -1 if the source does not have label
    flag = target_box[0].item()
    if flag < 0:
        return output_boxes, -1
    
    target_x1 = target_box[0].item()/width
    target_x2 = target_box[2].item()/width
    target_y1 = target_box[1].item()/height
    target_y2 = target_box[3].item()/height
    target_area = (target_y2-target_y1)*(target_x2-target_x1)
    
    for i in range(len(output_boxes)):
        box = output_boxes[i]
        score = output_scores[i]
        if score < 0.5:
            continue
        
        box_x1 = box[0].item()
        box_y1 = box[1].item()
        box_x2 = box[2].item()
        box_y2 = box[3].item()
        output_area = (box_y2-box_y1)*(box_x2-box_x1)
        
        merge_x1 = max([target_x1, box_x1])
        merge_x2 = min([target_x2, box_x2])
        merge_y1 = max([target_y1, box_y1])
        merge_y2 = min([target_y2, box_y2])
        
        merge_area = 0
        iou = 0.0
        if merge_y2-merge_y1>0 and merge_x2-merge_x1>0:
            merge_area = (merge_y2-merge_y1)*(merge_x2-merge_x1)
            iou = merge_area / (output_area+target_area-merge_area)
        
        if iou > best_iou:
            best_iou = iou
            best_box = box
    
#     print([target_x1, target_y1, target_x2, target_y2])
#     print(best_box)
#     print(best_iou)
    
    return best_box, best_iou
    

def plotting(train_score_list, train_loss_list, val_score_list, val_loss_list, model_path):
    df1 = pd.DataFrame(
        {'epoch': np.arange(len(train_loss_list)), 'train loss': train_loss_list, 'validation loss': val_loss_list})
    plt.plot('epoch', 'train loss', data=df1, marker='', color='skyblue', linewidth=2)
    plt.plot('epoch', 'validation loss', data=df1, marker='', color='olive', linewidth=2)
    plt.xlabel('epoch\n(a) Training and Validation Loss')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(model_path+'/loss.png')
    plt.clf()

    df2 = pd.DataFrame(
        {'epoch': np.arange(len(train_score_list)), 'train score': train_score_list,
         'validation score': val_score_list})
    plt.plot('epoch', 'train score', data=df2, marker='', color='skyblue', linewidth=2)
    plt.plot('epoch', 'validation score', data=df2, marker='', color='olive', linewidth=2)
    plt.xlabel('epoch\n(a) Training and Validation Score')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(model_path+'/score.png')
    plt.close()
