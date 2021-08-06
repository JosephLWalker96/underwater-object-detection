import argparse

import pandas as pd
import numpy as np
import cv2
import os
import torch
from torch.utils.data import DataLoader
from utils import collate_fn, get_test_transform
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


def generate_image_with_bbox(model, test_dataset, qr_df, path_to_images):
    print("iterating through the test images")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data_loader = DataLoader(test_dataset, shuffle=True, batch_size=4, pin_memory=True, collate_fn=collate_fn,
                             num_workers=4)
    for images, targets, image_ids in tqdm(data_loader):
        patch_size = image_ids.__len__()
        for i in range(patch_size):
            model.eval()
            output = model([images[i].to(device)])

            # converting bbox location into range (0, 1)
            boxes = output[0]['boxes'] / 512
            scores = output[0]['scores']

            # obtaining the original images
            idx = image_ids[i]
            records = qr_df[qr_df.index == idx]
            img_folder = path_to_images + "/" + str(records["Location"].values[0]) + "/" + str(
                records["Camera"].values[0])
            img_path = img_folder + "/" + str(records["File Name"].values[0])
            img = cv2.imread(img_path)

            # fitting bbox location in the original images
            for j in range(boxes.shape[0]):
                score = scores[j].item()
                if score < 0.5:
                    continue

                box = boxes[j]
                
                if not os.path.exists(path_to_images+'/labels'):
                    os.system('mkdir '+path_to_images+'/labels')
                
                with open(path_to_images+'/labels/'+str(records["Image"].values[0])+'.txt', 'a') as f:
                    xs = (box[0]+box[2])/2
                    ys = (box[1]+box[3])/2
                    w = box[2] - box[0]
                    h = box[3] - box[1]
                    line = '0 '+str(float(xs))+' '+str(float(ys))+' '+str(float(w))+' '+str(float(h))+'\n'
                    f.write(line)
                
                x1 = int(box[0] * img.shape[1])
                y1 = int(box[1] * img.shape[0])
                x2 = int(box[2] * img.shape[1])
                y2 = int(box[3] * img.shape[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 10)

            path_to_bbox_images = img_folder + "/images_with_bbox"
            if not os.path.exists(path_to_bbox_images):
                os.mkdir(path_to_bbox_images)
            filename = path_to_bbox_images + "/" + str(records["File Name"].values[0])
            cv2.imwrite(filename, img)


def run(args):
    # Input for Images Folder
    path_to_images = args.image_path

    # check whether the path exists
    if not os.path.exists(path_to_images):
        print("Test Image Path does not exist")
        raise FileNotFoundError

    if not os.path.exists(path_to_images + "/test_qr_labels.csv"):
        print("getting csv for testing data")
        generate_test_csv(path_to_images)

    print("loading " + path_to_images + "/test_qr_labels.csv")
    qr_df = pd.read_csv(path_to_images + "/test_qr_labels.csv")
    test_tf = get_test_transform()
    test_dataset = QRDatasets(path_to_images+'/test', qr_df, transforms=test_tf)

    # loading up the model
    model = None
    if os.path.exists("models/" + args.model):
        print("loading model")
        with open("models/" + args.model, 'rb') as f:
            model = torch.load("models/" + args.model)
    else:
        print("model does not exist")
        print("training a new model")
        args.image_path = args.train_image_path
        train.main(args)
        print("loading model")
        with open("models/" + args.model, 'rb') as f:
            model = torch.load("models/" + args.model)
    generate_image_with_bbox(model, test_dataset, qr_df, path_to_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--test_image_path', type=str)
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--label_path', type=str)
    parser.add_argument('--model', default='faster-rcnn', type=str)
    args = parser.parse_args()
    run(args)
