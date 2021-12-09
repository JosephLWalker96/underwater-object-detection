'''
    This script uses the qr detector for the WeChatCV Repo (https://github.com/WeChatCV/opencv_3rdparty/tree/wechat_qrcode).
    The code for img_transform  is modified from the source code from OpenCV WeChat QR Detection (https://github.com/opencv/opencv_contrib/tree/4.x/modules/wechat_qrcode).
'''
import cv2
import os
import pandas as pd
import numpy as np
import argparse
import tqdm


def img_transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    minInputSize = 400
    resizeRatio = np.sqrt(img.shape[1] * img.shape[0] / (minInputSize * minInputSize))
    target_w = int(img.shape[1] / resizeRatio)
    target_h = int(img.shape[0] / resizeRatio)

    input = cv2.resize(src=img, dsize=(target_h, target_w), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    input = cv2.dnn.blobFromImage(image=input, scalefactor=1.0 / 255, size=(input.shape[0], input.shape[1]),
                                  mean=(0.0, 0.0, 0.0), swapRB=False, crop=False, ddepth=cv2.CV_32F)

    return input


def detect(img_path: str, model: cv2.dnn):
    img = cv2.imread(img_path)
    input_img = img_transform(img)

    model.setInput(input_img)
    output = model.forward("detection_output")
    prob_scores = output[0][0]
    point = -1 * np.ones(4)  # x0 y0 x1 y1
    best_confidence = 0
    for prob_score in prob_scores:
        if prob_score[1] == 1 and prob_score[2] > 1E-5:
            confidence = prob_score[2]
            if confidence > best_confidence:
                best_confidence = confidence
                point[0] = prob_score[3]
                point[1] = prob_score[4]
                point[2] = prob_score[5]
                point[3] = prob_score[6]

    return point


def main(args):
    path_to_csv = os.path.join(args.dir_to_csv, args.project_name + '.csv')
    df = pd.DataFrame(columns=['img', 'x0', 'y0', 'x1', 'y1'])

    model = cv2.dnn.readNetFromCaffe("detect.prototxt", "detect.caffemodel")
    for file in tqdm.tqdm(os.listdir(args.path_to_dataset)):
        ext = file[-3:]
        if ext == 'JPG':
            img_path = os.path.join(args.path_to_dataset, file)
            point = detect(img_path, model)
            if point[0] != -1:
                df = df.append({
                    'img': file,
                    'x0': point[0],
                    'y0': point[1],
                    'x1': point[2],
                    'y1': point[3]
                }, ignore_index=True)

    df.to_csv(path_to_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # should be a dir to imgs
    parser.add_argument('--path_to_dataset', default='../full_resolution_images_bbox', type=str)
    parser.add_argument('--dir_to_csv', default='../full_resolution_images_bbox', type=str)
    parser.add_argument('--project_name', default='output', type=str)
    args = parser.parse_args()
    main(args)
