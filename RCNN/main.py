import argparse

import pandas as pd
import numpy as np
import cv2
import os
import torch
from torch.utils.data import DataLoader
from utils import collate_fn, get_test_transform, get_iou_score
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


def generate_image_with_bbox(path_to_output, model, test_dataset, qr_df, path_to_images, use_grayscale):
    print("iterating through the transform_test images")
    
    rslt_df = pd.DataFrame(columns=['img_path', 'img', 'xs', 'ys', 'w', 'h', 'iou_score'])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data_loader = DataLoader(test_dataset, shuffle=True, batch_size=1, pin_memory=True, collate_fn=collate_fn,num_workers=1)
    
    iou_scores = []
    with torch.no_grad():
        for images, targets, image_ids in tqdm(data_loader):
            patch_size = image_ids.__len__()
            for i in range(patch_size):
                model.eval()
                outputs = model([images[i].to(device)])
                target = targets[i]

                # converting bbox location into range (0, 1)
                outputs = outputs[0]
                boxes = outputs['boxes']
                scores = outputs['scores']
                labels = outputs['labels']
                outputs['boxes'] = outputs['boxes'] / 512

                # obtaining the original images
                idx = image_ids[i]
                records = qr_df[qr_df.index == idx]
                img_path = str(records["img_path"].values[0])

                img = cv2.imread(img_path)

                target_n = len(target['labels'])

                for j in range(target_n):
                    result_box, iou_score = get_iou_score(outputs, target, 512, 512, j)

                    if iou_score >= 0:
                        iou_scores.append(iou_score)
                        if result_box is not None:
                            img, rslt_df = draw_bbox(path_to_output, path_to_images, records, result_box, img, iou_score, rslt_df, target['labels'][j])
                    else:
                        for i in range(len(boxes)):
                            box = boxes[i]
                            score = scores[i]
                            label = labels[i]
                            if score < 0.5:
                                continue
                            img, rslt_df = draw_bbox(path_to_output, path_to_images, records, box, img, iou_score, rslt_df, label)

                    path_to_bbox_images = path_to_output + "/images_with_bbox"
                    if not os.path.exists(path_to_bbox_images):
                        os.mkdir(path_to_bbox_images)
                    filename = path_to_bbox_images + "/" + str(records["File Name"].values[0])
    #                 cv2.imwrite(filename, img)

    print('average transform_test iou scores = '+str(np.mean(iou_scores)))
    rslt_df.to_csv(path_to_output + "/labels/result_qr_labels.csv")
                

# fitting bbox location in the original images
def draw_bbox(path_to_output, path_to_images, records, box, img, iou_score, df, label):
    if not os.path.exists(path_to_output+'/labels'):
        os.system('mkdir '+path_to_output+'/labels')

    img_exts = str(records["File Name"].values[0]).split('.')

    with open(path_to_output+'/labels/'+img_exts[0]+'.txt', 'a') as f:
        xs = float((box[0]+box[2])/2)
        ys = float((box[1]+box[3])/2)
        w = float(box[2] - box[0])
        h = float(box[3] - box[1])
        line = str(int(label))+' '+str(float(xs))+' '+str(float(ys))+' '+str(float(w))+' '+str(float(h))+'\n'
        f.write(line)
        
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

    x1 = int(box[0] * img.shape[1])
    y1 = int(box[1] * img.shape[0])
    x2 = int(box[2] * img.shape[1])
    y2 = int(box[3] * img.shape[0])
    # red
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 10)
  
    # uncomment the following to get the original (target) box drawn with blue
    boxes = records[['x', 'y', 'w', 'h']].values 
    boxes[:, 2] = int((boxes[:, 0] + boxes[:, 2]))
    boxes[:, 3] = int((boxes[:, 1] + boxes[:, 3]))
    boxes[:, 0] = int(boxes[:, 0])
    boxes[:, 1] = int(boxes[:, 1])
    target_x1 = int(boxes[0][0])
    target_y1 = int(boxes[0][1])
    target_x2 = int(boxes[0][2])
    target_y2 = int(boxes[0][3])
#     print((target_x1, target_y1))
    # blue
    cv2.rectangle(img, (target_x1, target_y1), (target_x2, target_y2), (0, 255, 0), 10)
    
    return img, df

def main(path_to_output, path_to_images, path_to_labels, model_path, model_name):
    if not os.path.exists(path_to_output):
        os.mkdir(path_to_output)
        
    # Input for Images Folder
    args.image_path = path_to_images
    args.label_path = path_to_labels

    # check whether the path exists
    if not os.path.exists(path_to_images + "/test_qr_labels.csv"):
        print("getting csv for testing data")
        generate_test_csv(path_to_images)

    print("loading " + path_to_images + "/test_qr_labels.csv")
    qr_df = pd.read_csv(path_to_images + "/test_qr_labels.csv")
    test_tf = get_test_transform()
    test_dataset = QRDatasets(path_to_images+'/transform_test', qr_df, transforms=test_tf)

    # loading up the model
    model = None
    print(model_path + '/' + model_name)
    if os.path.exists(model_path + '/' + model_name):
        print("loading model")
        with open(model_path + '/' + model_name, 'rb') as f:
            model = torch.load(model_path + '/' + model_name)
    else:
        print("model does not exist")
        print("training a new model")
#         args.image_path = args.train_image_path
        train.main(args)
        print("loading model from "+model_path + '/' + model_name)
        with open(model_path + '/' + model_name, 'rb') as f:
            model = torch.load(model_path + '/' + model_name)
    generate_image_with_bbox(path_to_output, model, test_dataset, qr_df, path_to_images + '/transform_test', args.use_grayscale)
                
def run(args):
    dirs_to_images = []
    dirs_to_labels = []
    if args.exp_num == 'exp3' or args.exp_num == 'exp4':
        name_ls = ["HUA", "MOO", "RAI", "TAH", "TTR", "LL", "PAL"]
        for loc in name_ls:
            path_to_dir = os.path.join(args.path_to_dataset, args.exp_num)
            path_to_dir = os.path.join(path_to_dir, loc)
            path_to_images = os.path.join(path_to_dir, 'images')
            path_to_labels = os.path.join(path_to_dir, 'labels')
            model_path = os.path.join(path_to_dir, 'models')
            path_to_output = os.path.join(path_to_dir, 'rcnn')
#             os.system('rm -r '+model_path)
#             os.mkdir(model_path)
            main(path_to_output, path_to_images, path_to_labels, model_path, args.model)
    else:
        if args.exp_num is None:
            path_to_images = args.path_to_dataset + '/images'
            path_to_labels = args.path_to_dataset + '/labels'
            model_path = args.path_to_model
            path_to_output = args.path_to_dataset + '/rcnn'
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
    parser.add_argument('--path_to_model', default='../Complete_SUIT_Dataset', type=str)
    parser.add_argument('--exp_num', default='exp4', type=str)
    parser.add_argument('--transform_test',default=False, action='store_true')
#     parser.add_argument('--image_path', default='../Datasets/images', type=str)
#     parser.add_argument('--label_path', default='../Datasets/labels', type=str)
    parser.add_argument('--model', default='faster-rcnn', type=str)
#     parser.add_argument('--model', default='retinanet', type=str)
    parser.add_argument('--lr', default=0.02, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--step_size', default=5, type=int)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--early_stop', default=2, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--valid_ratio', default=0.2, type=float)
    parser.add_argument('--use_grayscale', default=False, action='store_true')
    args = parser.parse_args()
    run(args)


