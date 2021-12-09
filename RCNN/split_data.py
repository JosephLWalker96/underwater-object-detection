import argparse

import re
import pandas as pd
import numpy as np
import cv2
import glob, os
from tqdm import tqdm


'''
    This function is coming from datasets.py in yolov5,
    which is intended to get the corresponding txt position from given img_path
'''
def img2label_path(img_path):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    # print(sb.join(img_path.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt')
    return sb.join(img_path.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt'

'''
    This function is used to read the txt file.
    The txt file is one-line and the format for the txt should be:
    
    label, x, y, w, h

    @param txt_path: the path to the txt file
    @return: label, x, y, w, h    
'''
def read_txt(txt_path):
    x, y, w, h = -1, -1, -1, -1
    with open(txt_path, 'r') as f:
        #             print(txt_path)
        annot = f.readlines()
        # print(annot[0])
        annot_p = rr = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", annot[0])
        label, x, y, w, h = int(annot_p[0]), float(annot_p[1]), float(annot_p[2]), float(annot_p[3]), float(annot_p[4])

    # label is 1 or 2: 1 is SUIT, 2 is target
    return label + 2, x, y, w, h


'''
    This function is used to append a new row to qr_df
    @return qr_df: the dataframe that contains:
            "img_path": the original path to the image,
            "Location",
            "Camera",
            "File Name": the name of the image file,
            "Image": the name of the image file without file extension,
            "Image Width",
            "Image Height"
'''
def store_new_qr_entry(qr_df, cam_path, loc_name, cam_name, filename):
    #     print(cam_path+'/'+filename)
    img_name, img_ext = os.path.splitext(filename)
    if img_ext == ".JPG" or img_ext == ".PNG":
        img_path = os.path.abspath(cam_path + "/" + filename)
        img = np.array(cv2.imread(img_path))
        qr_df = qr_df.append({
            "img_path": img_path,
            "Location": loc_name,
            "Camera": cam_name,
            "File Name": filename,
            "Image": img_name,
            "Image Width": img.shape[1],
            "Image Height": img.shape[0]
        }, ignore_index=True)
    return qr_df


'''
    This function is used to append a new row to out_df
    @param row: the corresponding row to qr_df 
                (where it stores the information about the same image)
    @return out_df: the dataframe that contains:
            "img_path": the original path to the image,
            "Location",
            "Camera",
            "File Name": the name of the image file,
            "Image": the name of the image file without file extension,
            "Image Width",
            "Image Height",
            "x", "y", "w", "h"
    
'''
def store_new_out_entry(label, x, y, w, h, row, out_df):
    if not label == 0:
        width = int(row['Image Width'])
        height = int(row['Image Height'])

        xs = int(x * width)
        ys = int(y * height)

        w = int(w * width)
        h = int(h * height)

        x1 = xs - int(.5 * w)
        x2 = xs + int(.5 * w)
        y1 = ys - int(.5 * h)
        y2 = ys + int(.5 * h)

        x = x1
        y = y1

    out_df = out_df.append({
        "img_path": row['img_path'],
        "Location": row['Location'],
        "Camera": row['Camera'],
        "File Name": row['File Name'],
        "Image": row['Image'],
        "Image Width": row['Image Width'],
        "Image Height": row['Image Height'],
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "Labels": label
    }, ignore_index=True)

    return out_df


'''
    This function is used to obtain out_df
    @param path_to_images: The path to the image directory, i.e. /usr/.../Datasets/Image.
                           The image directory should look like:
                           Datasets/Image/
                                Location1/
                                    Camera/
                                        Focal Length/
                                            image1.JPG
                                            ...
                                Location2/
                                    Camera/
                                        image2.JPG
                                        ...
                                ...
                                HUA/
                                    JPEG/
                                        24mm/
                                            HUA_2021-05-27_15_JPEG_24mm_DSC_7299.JPG
                                            ...
                                PAL/
                                    ...
                                TTH/
                                    ...
                                ...
    @param path_to_bbox: The path to the annotation, i.e. /usr/.../Datasets/Annotation.
                         The annotation directory should look like:
                         Datasets/Annotation
                                SUIT/
                                    HUA_2021-05-27_15_JPEG_24mm_DSC_7298.txt
                                    ...
                                target/
                                    HUA_2021-05-27_15_JPEG_24mm_DSC_7298.txt
                                    ...
                           
    @return out_df: The Complete Dataframe that connects information of every images to every annotation. 
            The out_df contains:
            "img_path": the original path to the image,
            "Location",
            "Camera",
            "File Name": the name of the image file,
            "Image": the name of the image file without file extension,
            "Image Width",
            "Image Height",
            "x", "y", "w", "h"

'''
def get_qr_df(path_to_images, path_to_bbox):
    qr_df = pd.DataFrame(columns=["Location", "Camera", "File Name", "Image", "Image Width", "Image Height"])
    ## iterating via directory
    directory = path_to_images

    # getting the corresponding labels directory
    label_directory = path_to_images.replace('images', 'labels')
    if not os.path.exists(label_directory):
        os.mkdir(label_directory)

    print("Getting Images Info")
    for loc_name in tqdm(os.listdir(directory)):
        loc_path = directory + "/" + loc_name
        if os.path.isdir(loc_path):

            # creating the corresponding path for txt
            if not os.path.exists(label_directory + "/" + loc_name):
                os.mkdir(label_directory + "/" + loc_name)

            for cam_name in os.listdir(loc_path):
                cam_path = loc_path + "/" + cam_name
                if os.path.isdir(cam_path):

                    # creating the corresponding path for txt
                    if not os.path.exists(label_directory + "/" + loc_name + "/" + cam_name):
                        os.mkdir(label_directory + "/" + loc_name + "/" + cam_name)

                    for f_len_or_filename in os.listdir(cam_path):
                        if os.path.isdir(cam_path + '/' + f_len_or_filename):
                            f_len = f_len_or_filename

                            # creating the corresponding path for txt
                            if not os.path.exists(label_directory + "/" + loc_name + "/" + cam_name + '/' + f_len):
                                os.mkdir(label_directory + "/" + loc_name + "/" + cam_name + '/' + f_len)

                            for filename in os.listdir(cam_path + '/' + f_len):
                                qr_df = store_new_qr_entry(qr_df, cam_path + '/' + f_len, loc_name, cam_name, filename)
                        else:
                            filename = f_len_or_filename
                            qr_df = store_new_qr_entry(qr_df, cam_path, loc_name, cam_name, filename)

    #     print(len(qr_df))
    print("Getting SUIT and target Info")
    # setting up dataset
    out_df = pd.DataFrame(columns=["img_path", "Location", "Camera", "File Name",
                                   "Image", "Image Width", "Image Height",
                                   "x", "y", "w", "h", "Labels"])

    suit_dir = path_to_bbox + '/SUIT'
    target_dir = path_to_bbox + '/target'

    for idx, row in qr_df.iterrows():
        img_name = row["Image"]
        suit_path = suit_dir + '/' + img_name + '.txt'
        target_path = target_dir + '/' + img_name + '.txt'
        if os.path.exists(suit_path):
            label, x, y, w, h = read_txt(suit_path)
            out_df = store_new_out_entry(label, x, y, w, h, row, out_df)
        if os.path.exists(target_path):
            label, x, y, w, h = read_txt(target_path)
            out_df = store_new_out_entry(label, x, y, w, h, row, out_df)
        if not (os.path.exists(suit_path) or os.path.exists(target_path)):
            # label 0 means no bbox available
            out_df = store_new_out_entry(0, -1, -1, -1, -1, row, out_df)

    # print("writting csv for testing data to " + path_to_images + "/test_qr_labels.csv")
    return out_df


'''
    This function is used to find the out_df given by a series of image names
    The format for out_df is the following: 
        {"Location", "Camera", "File Name", "Image", "Image Width", "Image Height", "x", "y", "w", "h", "Labels"}
    @param df: The Complete Dataframe that connects information of every images to every annotation
'''
def get_df_by_name(df: pd.DataFrame, name_series:np.array):
    out_df = pd.DataFrame(columns=df.columns)
    for img_name in name_series:
        records = df.loc[df['Image'] == img_name]
        for idx, record in records.iterrows():
            out_df = out_df.append({
                "img_path": record['img_path'],
                "Location": record["Location"],
                "Camera": record["Camera"],
                "File Name": record["File Name"],
                "Image": record["Image"],
                "Image Width": record["Image Width"],
                "Image Height": record["Image Height"],
                "x": record["x"],
                "y": record["y"],
                "w": record["w"],
                "h": record["h"],
                "Labels": record["Labels"]
            }, ignore_index=True)
    return out_df


'''
    This function divides the sample randomly by images names into train, transform_test and val
    @param qr_df: The Complete Dataframe that connects information of every required image to its every annotation
    @param train_ratio: The ratio for train/all_data 
    @return: train_df, test_df, val_df that contains:
            "img_path": the original path to the image,
            "Location",
            "Camera",
            "File Name": the name of the image file,
            "Image": the name of the image file without file extension,
            "Image Width",
            "Image Height",
            "x", "y", "w", "h"
'''
def random_sample(qr_df: pd.DataFrame, train_ratio):
    image_columns = qr_df['Image'].unique()
    trainable_image_columns = qr_df[qr_df["Labels"] != 0]['Image'].unique()
    no_label_images_columns = np.array([img for img in image_columns if img not in trainable_image_columns])

    # getting train and validation df
    indices = np.arange(len(trainable_image_columns))
    train_size = int(float(train_ratio) * len(image_columns))
    
    train_val_indices = np.random.choice(indices, train_size, replace=False)
    train_indices = np.random.choice(train_val_indices, int(len(train_val_indices)*train_ratio), replace=False)
    val_indices = np.array([idx for idx in train_val_indices if idx not in train_indices])
    
    train_name_series = trainable_image_columns[train_indices]
    val_name_series = trainable_image_columns[val_indices]
    
    # getting transform_test indices
    test_indices = np.array([idx for idx in indices if idx not in train_val_indices])
    test_name_series = np.concatenate([trainable_image_columns[test_indices], no_label_images_columns], axis=None)

    train_df = get_df_by_name(qr_df, train_name_series)
    test_df = get_df_by_name(qr_df, test_name_series)
    val_df = get_df_by_name(qr_df, val_name_series)

    return train_df, test_df, val_df

'''
    This function divides the sample into train, transform_test and val for different experiments
'''
def getTrainTestVal(train_ratio, qr_df, train_on=None, test_on=None, exp_num='exp1'):
    # Split Data Ramdomly
    if exp_num is None or exp_num == 'exp1':

        train_df, test_df, val_df = random_sample(qr_df, train_ratio)
        print('train size = ' + str(len(train_df)))
        print('val size = ' + str(len(val_df)))
        print('transform_test size = ' + str(len(test_df)))

        return train_df, test_df, val_df
    # Train on (HUA, MOO, RAI,TAH, TTR) and transform_test on (LL,PAL).
    elif exp_num == 'exp2':
        new_data_df = qr_df[qr_df['Location'].str.contains("HUA|MOO|RAI|TAH|TTR")]
        old_data_df = qr_df[qr_df['Location'].str.contains("LL|PAL")]

        train_df1, val_df, train_df2 = random_sample(new_data_df, train_ratio)
        train_df = pd.concat([train_df1, train_df2])
        train_df.index = pd.RangeIndex(len(train_df.index))

        test_df = pd.concat(random_sample(old_data_df, train_ratio))
        test_df.index = pd.RangeIndex(len(test_df.index))

        return train_df, test_df, val_df

    # transform_test on the image data from one environment and train on all the other images
    # (for every environment, split 0.8:0.2 for training and testing)
    elif exp_num == 'exp3':
        selections = ["HUA", "MOO", "RAI", "TAH", "TTR", "LL", "PAL"]

        assert test_on in selections

        train_df = []
        val_df = []
        test_df = None

        for loc in selections:
            if not loc == test_on:
                df = qr_df[qr_df['Location'].str.contains(loc)]
                curr_train_df1, curr_val_df, curr_train_df2 = random_sample(df, train_ratio)
                curr_train_df = pd.concat([curr_train_df1, curr_train_df2])
                curr_train_df.index = pd.RangeIndex(len(curr_train_df.index))
                train_df.append(curr_train_df)
                val_df.append(curr_val_df)
            else:
                test_df = qr_df[qr_df['Location'].str.contains(loc)]

        train_df = pd.concat(train_df)
        val_df = pd.concat(val_df)
        test_df.index = pd.RangeIndex(len(test_df.index))
        train_df.index = pd.RangeIndex(len(train_df.index))
        val_df.index = pd.RangeIndex(len(val_df.index))

        return train_df, test_df, val_df
    # train on the image data from one environment and transform_test on all the other images
    # (for every environment, split 0.8:0.2 for training and testing)
    elif exp_num == 'exp4':
        selections = ["HUA", "MOO", "RAI", "TAH", "TTR", "LL", "PAL"]

        assert train_on in selections

        train_df = []
        val_df = []
        test_df = None

        for loc in selections:
            if not loc == train_on:
                test_df = qr_df[qr_df['Location'].str.contains(loc)]
            else:
                df = qr_df[qr_df['Location'].str.contains(loc)]
                curr_train_df1, curr_val_df, curr_train_df2 = random_sample(df, train_ratio)
                curr_train_df = pd.concat([curr_train_df1, curr_train_df2])
                curr_train_df.index = pd.RangeIndex(len(curr_train_df.index))
                train_df.append(curr_train_df)
                val_df.append(curr_val_df)

        train_df = pd.concat(train_df)
        val_df = pd.concat(val_df)
        test_df.index = pd.RangeIndex(len(test_df.index))
        train_df.index = pd.RangeIndex(len(train_df.index))
        val_df.index = pd.RangeIndex(len(val_df.index))

        return train_df, test_df, val_df
    # cross validation with each environment as different patch (unused, not modified to fit target detection)
    elif exp_num == 'exp5':
        train_df = []
        test_df = []
        val_df = []

        HUA_df = qr_df[qr_df['Location'].str.contains("HUA")]
        HUA_train_df, HUA_test_df, HUA_val_df = random_sample(HUA_df, train_ratio)
        train_df.append(HUA_train_df)
        test_df.append(HUA_test_df)
        val_df.append(HUA_val_df)

        MOO_df = qr_df[qr_df['Location'].str.contains("MOO")]
        MOO_train_df, MOO_test_df, MOO_val_df = random_sample(MOO_df, train_ratio)
        train_df.append(MOO_train_df)
        test_df.append(MOO_test_df)
        val_df.append(MOO_val_df)

        RAI_df = qr_df[qr_df['Location'].str.contains("RAI")]
        RAI_train_df, RAI_test_df, RAI_val_df = random_sample(RAI_df, train_ratio)
        train_df.append(RAI_train_df)
        test_df.append(RAI_test_df)
        val_df.append(RAI_val_df)

        TAH_df = qr_df[qr_df['Location'].str.contains("TAH")]
        TAH_train_df, TAH_test_df, TAH_val_df = random_sample(TAH_df, train_ratio)
        train_df.append(TAH_train_df)
        test_df.append(TAH_test_df)
        val_df.append(TAH_val_df)

        TTR_df = qr_df[qr_df['Location'].str.contains("TTR")]
        TTR_train_df, TTR_test_df, TTR_val_df = random_sample(TTR_df, train_ratio)
        train_df.append(TTR_train_df)
        test_df.append(TTR_test_df)
        val_df.append(TTR_val_df)

        LL_df = qr_df[qr_df['Location'].str.contains("LL")]
        LL_train_df, LL_test_df, LL_val_df = random_sample(LL_df, train_ratio)
        train_df.append(LL_train_df)
        test_df.append(LL_test_df)
        val_df.append(LL_val_df)

        PAL_df = qr_df[qr_df['Location'].str.contains("PAL")]
        PAL_train_df, PAL_test_df, PAL_val_df = random_sample(PAL_df, train_ratio)
        train_df.append(PAL_train_df)
        test_df.append(PAL_test_df)
        val_df.append(PAL_val_df)

        return train_df, test_df, val_df
    else:
        return None, None, None


def run():
    path_to_images = args.dataset_path + '/images'
    path_to_bbox = args.dataset_path + '/labels'
    train_ratio = args.train_ratio
    exp_num = args.exp_num

    if exp_num is None:
        path_to_save = args.dataset_path
    else:
        path_to_save = args.dataset_path + '/' + exp_num
        os.system('rm -r ' + path_to_save)
        os.mkdir(path_to_save)

    # if not args.rerun and os.path.exists(args.dataset_path + "/all_qr_labels.csv"):
    #     qr_df = pd.read_csv(args.dataset_path + "/all_qr_labels.csv")
    # else:
    qr_df = get_qr_df(path_to_images, path_to_bbox)
#     qr_df = qr_df.sample(frac=1).reset_index(drop=True)
    qr_df.to_csv(args.dataset_path + "/all_qr_labels.csv")

    if exp_num is None or exp_num == 'exp1' or exp_num == 'exp2':
        if exp_num is not None:
            path_to_img = args.dataset_path + '/' + exp_num + '/images'
            path_to_label = args.dataset_path + '/' + exp_num + '/labels'
        else:
            path_to_img = args.dataset_path + '/images'
            path_to_label = args.dataset_path + '/labels'

        if not os.path.exists(path_to_img):
            os.mkdir(path_to_img)
        if not os.path.exists(path_to_label):
            os.mkdir(path_to_label)

        if train_ratio == 0:
            save_to_path(qr_df, path_to_images, path_to_bbox, path_to_save, 'test_qr_labels.csv', 'transform_test')
        else:
            if exp_num is not None:
                train_df, test_df, val_df = getTrainTestVal(train_ratio, qr_df, exp_num=exp_num)
            else:
                train_df, test_df, val_df = getTrainTestVal(train_ratio, qr_df)
            save_to_path(train_df, path_to_images, path_to_bbox, path_to_save, 'train_qr_labels.csv', 'train')
            save_to_path(test_df, path_to_images, path_to_bbox, path_to_save, 'test_qr_labels.csv', 'transform_test')
            save_to_path(val_df, path_to_images, path_to_bbox, path_to_save, 'val_qr_labels.csv', 'val')
    elif exp_num == 'exp3':
        name_ls = ["HUA", "MOO", "RAI", "TAH", "TTR", "LL", "PAL"]
        for loc in tqdm(name_ls):
            path_to_dir = args.dataset_path + '/' + exp_num + '/' + loc
            path_to_img = path_to_dir + '/images'
            path_to_label = path_to_dir + '/labels'
            os.mkdir(path_to_dir)
            os.mkdir(path_to_img)
            os.mkdir(path_to_label)
            train_df, test_df, val_df = getTrainTestVal(train_ratio, qr_df, test_on=loc, exp_num=exp_num)
            save_to_path(train_df, path_to_images, path_to_bbox, path_to_save + "/" + loc, 'train_qr_labels.csv',
                         'train')
            save_to_path(test_df, path_to_images, path_to_bbox, path_to_save + "/" + loc, 'test_qr_labels.csv', 'transform_test')
            save_to_path(val_df, path_to_images, path_to_bbox, path_to_save + "/" + loc, 'val_qr_labels.csv', 'val')
    elif exp_num == 'exp4':
        name_ls = ["HUA", "MOO", "RAI", "TAH", "TTR", "LL", "PAL"]
        for loc in tqdm(name_ls):
            path_to_dir = args.dataset_path + '/' + exp_num + '/' + loc
            path_to_img = path_to_dir + '/images'
            path_to_label = path_to_dir + '/labels'
            os.mkdir(path_to_dir)
            os.mkdir(path_to_img)
            os.mkdir(path_to_label)
            train_df, test_df, val_df = getTrainTestVal(train_ratio, qr_df, train_on=loc, exp_num=exp_num)
            save_to_path(train_df, path_to_images, path_to_bbox, path_to_save + "/" + loc, 'train_qr_labels.csv',
                         'train')
            save_to_path(test_df, path_to_images, path_to_bbox, path_to_save + "/" + loc, 'test_qr_labels.csv', 'transform_test')
            save_to_path(val_df, path_to_images, path_to_bbox, path_to_save + "/" + loc, 'val_qr_labels.csv', 'val')
    elif exp_num == 'exp5':
        path_to_img = args.dataset_path + '/' + exp_num + '/images'
        path_to_label = args.dataset_path + '/' + exp_num + '/labels'
        os.mkdir(path_to_img)
        os.mkdir(path_to_label)
        train_df_ls, test_df_ls, val_df_ls = getTrainTestVal(train_ratio, qr_df, exp_num)
        name_ls = ["HUA", "MOO", "RAI", "TAH", "TTR", "LL", "PAL"]

        for i in range(7):
            name = name_ls[i]
            train_df = train_df_ls[i]
            val_df = val_df_ls[i]
            train_df = pd.concat([train_df, val_df])
            train_df.index = pd.RangeIndex(len(train_df.index))

            save_to_path(train_df, path_to_images, path_to_bbox, path_to_save, name + '_train_qr_labels.csv', 'train')
            save_to_path(val_df, path_to_images, path_to_bbox, path_to_save, name + '_val_qr_labels.csv', 'val')
            print(name + '_train_size = ' + str(len(train_df)))
            print(name + '_val_size = ' + str(len(val_df)))

        test_df = pd.concat(test_df_ls)
        test_df.index = pd.RangeIndex(len(test_df.index))
        save_to_path(test_df, path_to_images, path_to_bbox, path_to_save, 'test_qr_labels.csv', 'transform_test')
        print('test_size = ' + str(len(test_df)))


def save_to_path(df, path_to_images, path_to_bbox, path_to_save, csv_filename, mode):
    # if not os.path.exists(path_to_save + "/images/" + mode):
        # os.mkdir(path_to_save + "/images/" + mode)
    # if not os.path.exists(path_to_save + "/labels/" + mode):
    #     os.mkdir(path_to_save + "/labels/" + mode)

    df.to_csv(path_to_save + '/images/' + csv_filename)
    image_columns = df['Image'].unique()

    img_txt_path = path_to_save + '/' + mode + '.txt'
    os.system('touch ' + img_txt_path)

    if not os.path.exists(img_txt_path):
        os.system('touch ' + img_txt_path)
    else:
        os.system('rm ' + img_txt_path)

    for img_name in tqdm(image_columns, desc = 'saving %s files'%mode):
        records = df.loc[df['Image'] == img_name]
        img_path = str(records['img_path'].values[0])

        with open(img_txt_path, 'a') as img_txt:
            written_str = img_path +'\n'
            img_txt.write(written_str)

        txt_w_path = img2label_path(img_path)

        suit_r_path = path_to_bbox + '/SUIT/' + img_name + '.txt'
        target_r_path = path_to_bbox + '/target/' + img_name + '.txt'
        txt_r_paths = [suit_r_path, target_r_path]
        os.system('touch ' + txt_w_path)
        is_modify = False
        for idx, record in records.iterrows():
            label = record["Labels"]
            i = label - 1
            txt_r_path = txt_r_paths[i]

            with open(txt_w_path, 'a') as f:
                if not os.path.exists(txt_r_path):
                    continue
                _, x, y, w, h = read_txt(txt_r_path)
                if x < 0:
                    continue
                else:
                    written_str = '%d %f %f %f %f \n' % (label, x, y, w, h)
                    f.write(written_str)
                    # f.truncate()
                    is_modify = True
                f.close()
        if not is_modify and os.path.exists(txt_w_path):
            os.system('rm ' + txt_w_path)
    img_txt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='../Complete_SUIT_Dataset', type=str)
    parser.add_argument('--train_ratio', default=0.8, type=float)
    parser.add_argument('--exp_num', default=None, type=str, choices=['exp1', 'exp2', 'exp3', 'exp4'])
    args = parser.parse_args()
    run()
