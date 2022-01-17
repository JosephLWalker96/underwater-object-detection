import os

import cv2
import matplotlib
import yaml

matplotlib.use('Agg')


def collate_fn(batch):
    return tuple(zip(*batch))


def read_aug(input_ls):
    if input_ls is None:
        return None
    aug_ls = []
    for line in input_ls:
        aug, min_val, max_val = line.split(',')
        aug_ls.append((int(aug), float(min_val), float(max_val)))
    return aug_ls

def update_attrib(args):
    with open(args.yaml, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for k in config.keys():
        args.__setattr__(k, config[k])

    args.augment_list = read_aug(args.augment_list)

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

        write_rslt_txt(path_to_output, img_exts[0], label, xs, ys, w, h)
        df = update_df(df, records, xs, ys, w, h, iou_score, label)

    return df


def write_rslt_txt(path_to_output, img_name, label, xs, ys, w, h):
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
