import argparse

import pandas as pd
import numpy as np
import cv2
import os
import torch
from torch.utils.data import DataLoader

from wbf_ensemble import make_ensemble_predictions, run_wbf
from utils import collate_fn, get_iou_score, check_bbox, draw_bbox
from img_transform import get_test_transform
from QRDatasets import QRDatasets
from tqdm import tqdm
import train


def generate_test_csv(path_to_images):
    qr_df = pd.DataFrame(columns=["Location", "Camera", "File Name", "Image", "Image Width", "Image Height"])
    ## iterating via directory
    directory = path_to_images
    for loc_name in tqdm(os.listdir(directory)):
        loc_path = directory + "/" + loc_name
        if os.path.isdir(loc_path):
            for cam_name in os.listdir(loc_path):
                cam_path = loc_path + "/" + cam_name
                if os.path.isdir(cam_path):
                    for filename in os.listdir(cam_path):
                        img_name, img_ext = os.path.splitext(filename)
                        if img_ext == ".JPG" or img_ext == ".PNG":
                            img_path = cam_path + "/" + filename
                            img = np.array(cv2.imread(img_path))
                            qr_df = qr_df.append({
                                "Location": loc_name,
                                "Camera": cam_name,
                                "File Name": filename,
                                "Image": img_name,
                                "Image Width": img.shape[0],
                                "Image Height": img.shape[1]
                            }, ignore_index=True)
    # setting up dataset
    qr_df['x'] = -1
    qr_df['y'] = -1
    qr_df['w'] = -1
    qr_df['h'] = -1
    print("writting csv for testing data to " + path_to_images + "/test_qr_labels.csv")
    qr_df.to_csv(path_to_images + "/test_qr_labels.csv")


def test(path_to_output, model, test_dataset, qr_df):
    print("iterating through the test images")

    rslt_df = pd.DataFrame(columns=['img_path', 'img', 'xs', 'ys', 'w', 'h', 'iou_score'])
    data_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.test_batch_size, pin_memory=True,
                             collate_fn=collate_fn, num_workers=4)

    iou_scores = []
    with torch.no_grad():
        for images, targets, image_ids in tqdm(data_loader):
            model.eval()
            images = list(image.to(device) for image in images)
            predictions = make_ensemble_predictions(images, device, [model])

            for i, image in enumerate(images):
                target = targets[i]
                boxes, scores, labels = run_wbf(predictions, image_index=i)
                boxes = boxes.astype(np.int32).clip(min=0, max=512)/512
                outputs = {'boxes': boxes, 'scores': scores, 'labels': labels, 'IoU': -1 * np.ones(len(boxes))}

                target_n = len(target['labels'])
                for j in range(target_n):
                    result_box, iou_score = get_iou_score(outputs, target, 512, 512, j)

                    if iou_score > 0:
                        iou_scores.append(iou_score)

                # obtaining the original images
                idx = image_ids[i]
                records = qr_df[qr_df.index == idx]
                records = qr_df[qr_df['File Name'] == records['File Name'].values[0]]
                rslt_df = check_bbox(path_to_output, records, outputs, rslt_df)
                draw_bbox(path_to_output, args.path_to_dataset, records, boxes)

    print('average test iou scores = ' + str(np.mean(iou_scores)))
    rslt_df.to_csv(path_to_output + "/labels/result_qr_labels.csv")


def main(path_to_output, path_to_images, path_to_labels, model_path, model_name):
    if not os.path.exists(path_to_output):
        os.system('mkdir '+path_to_output)

    # Input for Images Folder
    args.image_path = path_to_images
    args.label_path = path_to_labels

    # check whether the path exists
    if not os.path.exists(path_to_images + "/test_qr_labels.csv"):
        print("getting csv for testing data")
        generate_test_csv(path_to_images)

    print("loading " + path_to_images + "/test_qr_labels.csv")
    qr_df = pd.read_csv(path_to_images + "/test_qr_labels.csv")
    test_tf = get_test_transform(args.test_transform)
    test_dataset = QRDatasets(args.path_to_dataset, qr_df, transforms=test_tf, use_grayscale=args.use_grayscale)

    # loading up the model
    model = None
    print(model_path + '/' + model_name)
    if os.path.exists(model_path + '/' + model_name):
        print("loading model")
        with open(model_path + '/' + model_name, 'rb') as f:
            model = torch.load(model_path + '/' + model_name, map_location=device)
    else:
        print("model does not exist")
        print("training a new model")
        #         args.image_path = args.train_image_path
        train.main(args)
        print("loading model from " + model_path + '/' + model_name)
        with open(model_path + '/' + model_name, 'rb') as f:
            model = torch.load(model_path + '/' + model_name, map_location=device)
    test(path_to_output, model, test_dataset, qr_df)


def run(args):
    dirs_to_images = []
    dirs_to_labels = []
    if args.exp_num == 'exp3' or args.exp_num == 'exp4':
        name_ls = ["HUA", "MOO", "RAI", "TAH", "TTR", "LL"]
        for loc in name_ls:
            path_to_dir = os.path.join(args.path_to_dataset, args.exp_num)
            path_to_dir = os.path.join(path_to_dir, loc)
            path_to_images = os.path.join(path_to_dir, 'images')
            path_to_labels = os.path.join(path_to_dir, 'labels')
            model_path = os.path.join(path_to_dir, 'models')
            path_to_output = os.path.join(path_to_dir, args.model)
            #             os.system('rm -r '+model_path)
            #             os.mkdir(model_path)
            main(path_to_output, path_to_images, path_to_labels, model_path, args.model)
    else:
        if args.exp_num is None:
            path_to_images = os.path.join(args.path_to_dataset, 'images')
            path_to_labels = os.path.join(args.path_to_dataset, 'labels')
            model_path = os.path.join(args.path_to_dataset, 'models')
            path_to_output = os.path.join(args.path_to_dataset, 'rcnn')
        else:
            path_to_images = args.path_to_dataset + '/' + args.exp_num + '/images'
            path_to_labels = args.path_to_dataset + '/' + args.exp_num + '/labels'
            model_path = args.path_to_dataset + '/' + args.exp_num + '/models'
            path_to_output = args.path_to_dataset + '/' + args.exp_num + '/rcnn'
        #         os.system('rm -r '+model_path)
        #         os.mkdir(model_path)
        main(path_to_output, path_to_images, path_to_labels, model_path, args.model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_dataset', default='../Complete_SUIT_Dataset', type=str)
    parser.add_argument('--exp_num', default='exp4', type=str)

    parser.add_argument('--model', default='faster-rcnn', type=str)
    parser.add_argument('--lr', default=0.0005, type=float)

    # parser.add_argument('--model', default='faster-rcnn-mobilenet', type=str)
    # parser.add_argument('--lr', default=0.001, type=float)

    #     parser.add_argument('--model', default='retinanet', type=str)

    parser.add_argument('--train_transform', default='intensive', type=str,
                        choices=['color_correction', 'default', 'intensive', 'no_transform'])
    parser.add_argument('--test_transform', default='no_transform', type=str,
                        choices=['color_correction', 'default', 'intensive', 'no_transform'])

    parser.add_argument('--adam', default=True, action='store_true')

    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--step_size', default=5, type=int)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--early_stop', default=3, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--test_batch_size', default=16, type=int)
    parser.add_argument('--valid_ratio', default=0.2, type=float)
    parser.add_argument('--use_grayscale', default=False, action='store_true')
    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    run(args)
