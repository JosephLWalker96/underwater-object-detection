import argparse

import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm


def generate_csv(path_to_images, path_to_bbox):
    qr_df = pd.DataFrame(columns=["Location", "Camera", "File Name", "Image", "Image Width", "Image Height"])
    ## iterating via directory
    directory = path_to_images
    print("Getting Images Info")
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
                                "Image Width": img.shape[1],
                                "Image Height": img.shape[0]
                            }, ignore_index=True)

    print("Getting SUIT Info")
    # setting up dataset
    qr_df['x'] = -1
    qr_df['y'] = -1
    qr_df['w'] = -1
    qr_df['h'] = -1

    directory = path_to_bbox
    for bboxes_txt in tqdm(os.listdir(directory)):
        txt_path = directory + "/" + bboxes_txt
        img_name = os.path.splitext(bboxes_txt)[0]

        # this image's bbox have two lines of data
        if img_name == 'DSC_6775':
            continue

        if qr_df["Image"][qr_df["Image"] == img_name].count() == 0:
            continue

        coord = np.loadtxt(txt_path)

        records = qr_df.loc[qr_df["Image"] == img_name]
        if qr_df["Image"][qr_df["Image"] == img_name].count() > 1:
            # records = records[0]
            continue

        width = int(records['Image Width'])
        height = int(records['Image Height'])

        xs = int(coord[1] * width)
        ys = int(coord[2] * height)

        w = int(coord[3] * width)
        h = int(coord[4] * height)

        x1 = xs - int(.5 * w)
        x2 = xs + int(.5 * w)

        y1 = ys - int(.5 * h)
        y2 = ys + int(.5 * h)

        qr_df.loc[qr_df["Image"] == img_name, 'x'] = x1
        qr_df.loc[qr_df["Image"] == img_name, 'y'] = y1
        qr_df.loc[qr_df["Image"] == img_name, 'w'] = w
        qr_df.loc[qr_df["Image"] == img_name, 'h'] = h

    # print("writting csv for testing data to " + path_to_images + "/test_qr_labels.csv")
    # qr_df.to_csv(path_to_images + "/test_qr_labels.csv")
    return qr_df


'''
    This function is used to group the dataset into different color temperature
    Regular: LL_109-110/NikonD780 + LL_115-116/NikonD780 + LL_125-126/NikonD7000_18mm #214
    Regular (Bright): PAL_FR36_2020-12-13/NikonD780 #55
    Cyan: LL_119-120/NikonD7000_18mm +LL_121-122/NikonD7000_18mm #82
    Blue: PAL_FR13_2020-12-13/* + PAL_FR36_2020-12-13/NikonD7000_18mm + PAL_FR36_2020-12-13/NikonD7000_55mm #104
'''


def split_data(qr_df):
    # Regular
    regulars = qr_df[
        (((qr_df['Location'] == 'LL_109-110') | (qr_df['Location'] == 'LL_115-116')) & (qr_df['Camera'] == 'NikonD780'))
        | ((qr_df['Location'] == 'LL_125-126') & (qr_df['Camera'] == 'NikonD7000_18mm'))]
    bright_regulars = qr_df[(qr_df['Location'] == 'PAL_FR36_2020-12-13') & (qr_df['Camera'] == 'NikonD780')]
    cyan = qr_df[((qr_df['Location'] == 'LL_119-120') | (qr_df['Location'] == 'LL_121-122'))
                     & (qr_df['Camera'] == 'NikonD7000_18mm')]
    blue = qr_df[(qr_df['Location'] == 'PAL_FR13_2020-12-13')
                     | ((qr_df['Location'] == 'PAL_FR36_2020-12-13') & (qr_df['Camera'] == 'NikonD7000_18mm'))
                     | ((qr_df['Location'] == 'PAL_FR36_2020-12-13') & (qr_df['Camera'] == 'NikonD7000_55mm'))]
    return regulars, bright_regulars, cyan, blue


def getTrainAndTest(train_ratio, qr_df, isRamdom=True):
    if isRamdom:
        train_size = int(float(train_ratio) * qr_df.__len__())
        print("the training size is " + str(train_size))
        print("the testing size is " + str(qr_df.__len__() - train_size))

        trainable_qr_df = qr_df[qr_df["x"] != -1]
        if train_size > trainable_qr_df.__len__():
            print("Please use smaller training size")
            raise ValueError
        train_df = trainable_qr_df[:train_size]
        train_df.index = pd.RangeIndex(len(train_df.index))
        train_indices = train_df.index
        test_df = qr_df[~qr_df.index.isin(train_indices)]
        test_df.index = pd.RangeIndex(len(test_df.index))
        return train_df, test_df
    else:
        regulars, bright_regulars, cyan, blue = split_data(qr_df)
        trainable_regulars = regulars[regulars["x"] != -1]
        trainable_indices = trainable_regulars.index
        untrainable_regulars = regulars[~regulars.index.isin(trainable_indices)]
        print("trainable size: "+str(trainable_regulars.__len__()))
        train_df = trainable_regulars
        test_df = pd.concat([untrainable_regulars,bright_regulars, cyan, blue])
        train_df.index = pd.RangeIndex(len(train_df.index))
        test_df.index = pd.RangeIndex(len(test_df.index))
        return train_df, test_df


def run(args):
    path_to_images = args.image_path
    path_to_bbox = args.label_path
    train_ratio = args.train_ratio

    path_to_train = path_to_images + '/../train_images'
    path_to_test = path_to_images + '/../test_images'

    qr_df = generate_csv(path_to_images, path_to_bbox)
    qr_df = qr_df.sample(frac=1).reset_index(drop=True)

    train_df, test_df = getTrainAndTest(train_ratio, qr_df, isRamdom=False)

    os.mkdir(path_to_train)
    os.mkdir(path_to_test)

    save_to_path(train_df, path_to_images, path_to_train, 'train_qr_labels.csv')
    save_to_path(test_df, path_to_images, path_to_test, 'test_qr_labels.csv')


def save_to_path(df, path_to_images, path_to_save, csv_filename):
    df.to_csv(path_to_save + '/' + csv_filename)
    for idx in range(df.__len__()):
        records = df[df.index == idx]
        # print(records)
        img_r_path = path_to_images + "/" + str(records["Location"].values[0]) + "/" + \
                     str(records["Camera"].values[0]) + "/" + str(records["File Name"].values[0])
        img_w_path = path_to_save + "/" + str(records["Location"].values[0]) + "/" + \
                     str(records["Camera"].values[0]) + "/" + str(records["File Name"].values[0])
        # img = cv2.imread(img_r_path)
        # with open(img_w_path, 'wb') as f:
        #     cv2.imwrite(img, img_w_path)
        if not os.path.exists(path_to_save + "/" + str(records["Location"].values[0])):
            os.system('mkdir ' + path_to_save + "/" + str(records["Location"].values[0]))
        if not os.path.exists(
                path_to_save + "/" + str(records["Location"].values[0]) + "/" + str(records["Camera"].values[0])):
            os.system('mkdir ' + path_to_save + "/" + str(records["Location"].values[0]) + "/" + str(
                records["Camera"].values[0]))
        os.system('mv ' + img_r_path + ' ' + img_w_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='../Datasets/Image', type=str)
    parser.add_argument('--label_path', default='../Datasets/bboxes_for_SUIT_images_2021-05-10-02-29-37', type=str)
    parser.add_argument('--train_ratio', default=0.76, type=str)
    args = parser.parse_args()
    run(args)
