import os

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def collate_fn(batch):
    return tuple(zip(*batch))


def get_iou_score(output, target, width, height, target_idx):
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
            return output_boxes, -1

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
                output['IoU'][i] = max(output['IoU'][i], -1)
            else:
                output['IoU'][i] = max(output['IoU'][i], iou)

        if iou > best_iou:
            best_iou = iou
            best_box = box

    return best_box, best_iou


def plotting(train_score_list, train_loss_list, val_score_list, val_loss_list, model_path):
    df1 = pd.DataFrame(
        {'epoch': np.arange(len(train_loss_list)), 'train loss': train_loss_list, 'validation loss': val_loss_list})
    plt.plot('epoch', 'train loss', data=df1, marker='', color='skyblue', linewidth=2)
    plt.plot('epoch', 'validation loss', data=df1, marker='', color='olive', linewidth=2)
    plt.xlabel('epoch\n(a) Training and Validation Loss')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(model_path + '/loss.png')
    plt.clf()

    df2 = pd.DataFrame(
        {'epoch': np.arange(len(train_score_list)), 'train score': train_score_list,
         'validation score': val_score_list})
    plt.plot('epoch', 'train score', data=df2, marker='', color='skyblue', linewidth=2)
    plt.plot('epoch', 'validation score', data=df2, marker='', color='olive', linewidth=2)
    plt.xlabel('epoch\n(a) Training and Validation Score')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(model_path + '/score.png')
    plt.close()


def check_bbox(path_to_output, records, outputs, df):
    if not os.path.exists(path_to_output + '/labels'):
        os.system('mkdir ' + path_to_output + '/labels')

    n = len(outputs['boxes'])

    for idx in range(n):
        box = outputs['boxes'][idx]
        label = outputs['labels'][idx]
        iou_score = outputs['IoU'][idx]

        img_exts = str(records["File Name"].values[0]).split('.')
        xs = float((box[0] + box[2]) / 2)
        ys = float((box[1] + box[3]) / 2)
        w = float(box[2] - box[0])
        h = float(box[3] - box[1])

        write_txt(path_to_output, img_exts[0], label, xs, ys, w, h)
        df = update_df(df, records, xs, ys, w, h, iou_score, label)

    return df


def write_txt(path_to_output, img_name, label, xs, ys, w, h):
    with open(path_to_output + '/labels/' + img_name + '.txt', 'a') as f:
        line = str(int(label)) + ' ' + str(float(xs)) + ' ' + str(float(ys)) + ' ' + str(float(w)) + ' ' + str(
            float(h)) + '\n'
        f.write(line)
    f.close()


def update_df(df, records, xs, ys, w, h, iou_score, label):
    df = df.append({
        'img_path': str(records["img_path"].values[0]),
        'img': str(records["File Name"].values[0]),
        'xs': xs,
        'ys': ys,
        'w': w,
        'h': h,
        'iou_score': iou_score,
        'label': int(label)
    }, ignore_index=True)
    return df


def draw_bbox(path_to_output, path_to_dataset, records, boxes):
    if not os.path.exists(path_to_output + '/labels'):
        os.system('mkdir ' + path_to_output + '/labels')

    img = cv2.imread(os.path.join(path_to_dataset, str(records["img_path"].values[0])))

    for box in boxes:
        x1 = int(box[0] * img.shape[1])
        y1 = int(box[1] * img.shape[0])
        x2 = int(box[2] * img.shape[1])
        y2 = int(box[3] * img.shape[0])
        # red
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 10)

    # uncomment the following to get the original (target) box drawn with green
    target_boxes = records[['x', 'y', 'w', 'h']].values
    target_boxes[:, 2] = (target_boxes[:, 0] + target_boxes[:, 2])
    target_boxes[:, 3] = (target_boxes[:, 1] + target_boxes[:, 3])
    target_boxes[:, 0] = target_boxes[:, 0]
    target_boxes[:, 1] = target_boxes[:, 1]

    for target_box in target_boxes:
        target_x1 = int(target_box[0])
        target_y1 = int(target_box[1])
        target_x2 = int(target_box[2])
        target_y2 = int(target_box[3])
        cv2.rectangle(img, (target_x1, target_y1), (target_x2, target_y2), (0, 255, 0), 10)

    img_dir = os.path.join(path_to_output, "images_with_bbox")
    filename = os.path.join(img_dir, str(records["File Name"].values[0]))
    cv2.imwrite(filename, img)
