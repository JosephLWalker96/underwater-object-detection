import argparse

import re
import pandas as pd
import numpy as np
import cv2
import glob,os
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
                    for f_len_or_filename in os.listdir(cam_path):
                        if os.path.isdir(cam_path+'/'+f_len_or_filename):
                            f_len = f_len_or_filename
                            for filename in os.listdir(cam_path+'/'+f_len):
                                qr_df = store_new_qr_entry(qr_df, cam_path+'/'+f_len, loc_name, cam_name, filename)
                        else:
                            filename = f_len_or_filename
                            qr_df = store_new_qr_entry(qr_df, cam_path, loc_name, cam_name, filename)
    
#     print(len(qr_df))
    print("Getting SUIT Info")
    # setting up dataset
    qr_df['x'] = -1
    qr_df['y'] = -1
    qr_df['w'] = -1
    qr_df['h'] = -1

    directory = path_to_bbox
    for bboxes_txt in tqdm(os.listdir(path_to_bbox)):
#         print(bboxes_txt)
        txt_path = directory + "/" + bboxes_txt
        img_name = os.path.splitext(bboxes_txt)[0]
#         coord = np.loadtxt(txt_path)

        label,x,y,w,h = 0, 0, 0, 0, 0
        with open(txt_path,'r') as f:
#             print(txt_path)
            annot = f.readlines()
            #print(annot[0])
            annot_p = rr = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", annot[0])
            label,x,y,w,h = int(annot_p[0]),float(annot_p[1]),float(annot_p[2]),float(annot_p[3]),float(annot_p[4])
        
        records = qr_df.loc[qr_df["Image"] == img_name]

        width = int(records['Image Width'].values[0])
        height = int(records['Image Height'].values[0])

        xs = int(x * width)
        ys = int(y * height)

        w = int(w * width)
        h = int(h * height)

        x1 = xs - int(.5 * w)
        x2 = xs + int(.5 * w)

        y1 = ys - int(.5 * h)
        y2 = ys + int(.5 * h)

        qr_df.loc[qr_df["Image"] == img_name, 'x'] = x1
        qr_df.loc[qr_df["Image"] == img_name, 'y'] = y1
        qr_df.loc[qr_df["Image"] == img_name, 'w'] = w
        qr_df.loc[qr_df["Image"] == img_name, 'h'] = h

    # print("writting csv for testing data to " + path_to_images + "/test_qr_labels.csv")
    return qr_df


def store_new_qr_entry(qr_df, cam_path, loc_name, cam_name, filename):
#     print(cam_path+'/'+filename)
    img_name, img_ext = os.path.splitext(filename)
    if img_ext == ".JPG" or img_ext == ".PNG":
        img_path = cam_path + "/" + filename
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
    
def get_train_val_test(qr_df, train_ratio):
    train_size = int(float(train_ratio) * qr_df.__len__())
    trainable_qr_df = qr_df[qr_df["x"] != -1]
    if train_size > trainable_qr_df.__len__():
        print("Please use smaller training size")
        raise ValueError

    # getting the test df
    trainable_qr_df = trainable_qr_df.sample(frac=1).reset_index(drop=True) #shuffle
    train_and_val_df = trainable_qr_df[:train_size]
    test_df = qr_df[~qr_df.index.isin(train_and_val_df.index)]
    test_df.index = pd.RangeIndex(len(test_df.index))

    # getting val_df
    train_and_val_df = train_and_val_df.sample(frac=1).reset_index(drop=True) #shuffle
    train_size = int(float(train_ratio) * train_and_val_df.__len__())
    train_df = train_and_val_df[:train_size]
    val_df = train_and_val_df[~train_and_val_df.index.isin(train_df.index)]
    
    train_df.index = pd.RangeIndex(len(train_df.index))
    val_df.index = pd.RangeIndex(len(val_df.index))
    
    return train_df, test_df, val_df
    
def getTrainAndTest(train_ratio, qr_df, train_on = None, test_on=None, exp_num = 'exp1'):
    # Split Data Ramdomly
    if exp_num is None or exp_num == 'exp1':
        
        train_df, test_df, val_df = get_train_val_test(qr_df, train_ratio)
        print('train size = '+str(len(train_df)))
        print('val size = '+str(len(val_df)))
        print('test size = '+str(len(test_df)))
        
        return train_df, test_df, val_df
    # Train on (HUA, MOO, RAI,TAH, TTR) and test on (LL,PAL).
    elif exp_num == 'exp2':
        new_data_df = qr_df[qr_df['Location'].str.contains("HUA|MOO|RAI|TAH|TTR")]
        old_data_df = qr_df[qr_df['Location'].str.contains("LL|PAL")]

        train_and_val_df = new_data_df[new_data_df['x']!=-1]

        untrainable_new_dataset_df = new_data_df[~new_data_df.index.isin(train_and_val_df.index)]
        test_df = pd.concat([old_data_df, untrainable_new_dataset_df])
        test_df.index = pd.RangeIndex(len(test_df.index))

        train_and_val_df = train_and_val_df.sample(frac=1).reset_index(drop=True) #shuffle
        train_size = int(float(train_ratio) * train_and_val_df.__len__())
        train_df = train_and_val_df[:train_size]
        val_df = train_and_val_df[~train_and_val_df.index.isin(train_df.index)]
        train_df.index = pd.RangeIndex(len(train_df.index))
        val_df.index = pd.RangeIndex(len(val_df.index))

        print('train size = '+str(len(train_df)))
        print('val size = '+str(len(val_df)))
        print('test size = '+str(len(test_df)))

        return train_df, test_df, val_df
    
    # Using leave-one-out cross validation, 
    # test on the image data from one environment and train on all the other images 
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
                train_and_val_df = df[df['x']!=-1]
                train_and_val_df = train_and_val_df.sample(frac=1).reset_index(drop=True) #shuffle
                train_size = int(float(train_ratio) * train_and_val_df.__len__())
                curr_train_df = train_and_val_df[:train_size]
                train_df.append(curr_train_df)
                val_df.append(train_and_val_df[~train_and_val_df.index.isin(curr_train_df.index)])
            else:
                test_df = qr_df[qr_df['Location'].str.contains(loc)]
        
        train_df = pd.concat(train_df)
        val_df = pd.concat(val_df)
        test_df.index = pd.RangeIndex(len(test_df.index))
        train_df.index = pd.RangeIndex(len(train_df.index))
        val_df.index = pd.RangeIndex(len(val_df.index))
        
        return train_df, test_df, val_df
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
                train_and_val_df = df[df['x']!=-1]
                train_and_val_df = train_and_val_df.sample(frac=1).reset_index(drop=True) #shuffle
                train_size = int(float(train_ratio) * train_and_val_df.__len__())
                curr_train_df = train_and_val_df[:train_size]
                train_df.append(curr_train_df)
                val_df.append(train_and_val_df[~train_and_val_df.index.isin(curr_train_df.index)])
        
        train_df = pd.concat(train_df)
        val_df = pd.concat(val_df)
        test_df.index = pd.RangeIndex(len(test_df.index))
        train_df.index = pd.RangeIndex(len(train_df.index))
        val_df.index = pd.RangeIndex(len(val_df.index))
        
        return train_df, test_df, val_df
    # cross validation
    elif exp_num == 'exp5':
        train_df = []
        test_df = []
        val_df = []
        
        HUA_df = qr_df[qr_df['Location'].str.contains("HUA")]
        HUA_train_df, HUA_test_df, HUA_val_df = get_train_val_test(HUA_df, train_ratio)
        train_df.append(HUA_train_df)
        test_df.append(HUA_test_df)
        val_df.append(HUA_val_df)
        
        MOO_df = qr_df[qr_df['Location'].str.contains("MOO")]
        MOO_train_df, MOO_test_df, MOO_val_df = get_train_val_test(MOO_df, train_ratio)
        train_df.append(MOO_train_df)
        test_df.append(MOO_test_df)
        val_df.append(MOO_val_df)
        
        RAI_df = qr_df[qr_df['Location'].str.contains("RAI")]
        RAI_train_df, RAI_test_df, RAI_val_df = get_train_val_test(RAI_df, train_ratio)
        train_df.append(RAI_train_df)
        test_df.append(RAI_test_df)
        val_df.append(RAI_val_df)
        
        TAH_df = qr_df[qr_df['Location'].str.contains("TAH")]
        TAH_train_df, TAH_test_df, TAH_val_df = get_train_val_test(TAH_df, train_ratio)
        train_df.append(TAH_train_df)
        test_df.append(TAH_test_df)
        val_df.append(TAH_val_df)
        
        TTR_df = qr_df[qr_df['Location'].str.contains("TTR")]
        TTR_train_df, TTR_test_df, TTR_val_df = get_train_val_test(TTR_df, train_ratio)
        train_df.append(TTR_train_df)
        test_df.append(TTR_test_df)
        val_df.append(TTR_val_df)
        
        LL_df = qr_df[qr_df['Location'].str.contains("LL")]
        LL_train_df, LL_test_df, LL_val_df = get_train_val_test(LL_df, train_ratio)
        train_df.append(LL_train_df)
        test_df.append(LL_test_df)
        val_df.append(LL_val_df)
        
        PAL_df = qr_df[qr_df['Location'].str.contains("PAL")]
        PAL_train_df, PAL_test_df, PAL_val_df = get_train_val_test(PAL_df, train_ratio)
        train_df.append(PAL_train_df)
        test_df.append(PAL_test_df)
        val_df.append(PAL_val_df)
        
        return train_df, test_df, val_df
    else:
        return None, None, None


def run(args):
    path_to_images = args.dataset_path+'/All_SUIT_images'
    path_to_bbox = args.dataset_path+'/All_SUIT_annotations'
    train_ratio = args.train_ratio
    exp_num = args.exp_num

    if exp_num is None:
        path_to_save = args.dataset_path
    else:
        path_to_save = args.dataset_path+'/'+exp_num
        os.system('rm -r '+path_to_save)
        os.mkdir(path_to_save)

    qr_df = generate_csv(path_to_images, path_to_bbox)
    qr_df = qr_df.sample(frac=1).reset_index(drop=True)
    qr_df.to_csv(args.dataset_path+"/all_qr_labels.csv")

    if exp_num is None or exp_num == 'exp1' or exp_num == 'exp2':
        if exp_num is not None:
            path_to_img = args.dataset_path+'/'+exp_num+'/images'
            path_to_label = args.dataset_path+'/'+exp_num+'/labels'
        else:
            path_to_img = args.dataset_path + '/images'
            path_to_label = args.dataset_path + '/labels'
        os.mkdir(path_to_img)
        os.mkdir(path_to_label)
        train_df, test_df, val_df = getTrainAndTest(train_ratio, qr_df, exp_num)
        save_to_path(train_df, path_to_images, path_to_bbox, path_to_save, 'train_qr_labels.csv', 'train')
        save_to_path(test_df, path_to_images, path_to_bbox, path_to_save, 'test_qr_labels.csv', 'test')
        save_to_path(val_df, path_to_images, path_to_bbox, path_to_save, 'val_qr_labels.csv', 'val')
    elif exp_num == 'exp3':
        name_ls = ["HUA", "MOO", "RAI", "TAH", "TTR", "LL", "PAL"]
        for loc in tqdm(name_ls):
            path_to_dir = args.dataset_path+'/'+exp_num+'/'+loc
            path_to_img = path_to_dir+'/images'
            path_to_label = path_to_dir+'/labels'
            os.mkdir(path_to_dir)
            os.mkdir(path_to_img)
            os.mkdir(path_to_label)
            train_df, test_df, val_df = getTrainAndTest(train_ratio, qr_df, test_on = loc, exp_num = exp_num)
            save_to_path(train_df, path_to_images, path_to_bbox, path_to_save+"/"+loc, 'train_qr_labels.csv', 'train')
            save_to_path(test_df, path_to_images, path_to_bbox, path_to_save+"/"+loc, 'test_qr_labels.csv', 'test')
            save_to_path(val_df, path_to_images, path_to_bbox, path_to_save+"/"+loc, 'val_qr_labels.csv', 'val')
    elif exp_num == 'exp4':
        name_ls = ["HUA", "MOO", "RAI", "TAH", "TTR", "LL", "PAL"]
        for loc in tqdm(name_ls):
            path_to_dir = args.dataset_path+'/'+exp_num+'/'+loc
            path_to_img = path_to_dir+'/images'
            path_to_label = path_to_dir+'/labels'
            os.mkdir(path_to_dir)
            os.mkdir(path_to_img)
            os.mkdir(path_to_label)
            train_df, test_df, val_df = getTrainAndTest(train_ratio, qr_df, train_on = loc, exp_num = exp_num)
            save_to_path(train_df, path_to_images, path_to_bbox, path_to_save+"/"+loc, 'train_qr_labels.csv', 'train')
            save_to_path(test_df, path_to_images, path_to_bbox, path_to_save+"/"+loc, 'test_qr_labels.csv', 'test')
            save_to_path(val_df, path_to_images, path_to_bbox, path_to_save+"/"+loc, 'val_qr_labels.csv', 'val')    
    elif exp_num == 'exp5':
        path_to_img = args.dataset_path+'/'+exp_num+'/images'
        path_to_label = args.dataset_path+'/'+exp_num+'/labels'
        os.mkdir(path_to_img)
        os.mkdir(path_to_label)
        train_df_ls, test_df_ls, val_df_ls = getTrainAndTest(train_ratio, qr_df, exp_num)
        name_ls = ["HUA", "MOO", "RAI", "TAH", "TTR", "LL", "PAL"]
        
        for i in range(7):
            name = name_ls[i]
            train_df = train_df_ls[i]
            val_df = val_df_ls[i]
            train_df = pd.concat([train_df, val_df])
            train_df.index = pd.RangeIndex(len(train_df.index))
            
            save_to_path(train_df, path_to_images, path_to_bbox, path_to_save, name+'_train_qr_labels.csv', 'train')
            save_to_path(val_df, path_to_images, path_to_bbox, path_to_save, name+'_val_qr_labels.csv', 'val')
            print(name+'_train_size = '+ str(len(train_df)))
            print(name+'_val_size = '+ str(len(val_df)))
        
        test_df = pd.concat(test_df_ls)
        test_df.index = pd.RangeIndex(len(test_df.index))
        save_to_path(test_df, path_to_images, path_to_bbox, path_to_save, 'test_qr_labels.csv', 'test')
        print('test_size = '+ str(len(test_df)))

def save_to_path(df, path_to_images, path_to_bbox, path_to_save, csv_filename, mode):
    if not os.path.exists(path_to_save+ "/images/"+mode):
        os.mkdir(path_to_save+ "/images/"+mode)
    if not os.path.exists(path_to_save+ "/labels/"+mode):
        os.mkdir(path_to_save+ "/labels/"+mode)
    df.to_csv(path_to_save + '/images/' + csv_filename)
    for idx in range(df.__len__()):
        records = df[df.index == idx]
        
        img_name = str(records["Image"].values[0])
        img_r_path = str(records["img_path"].values[0])
        img_w_path = path_to_save + "/images/" + mode
#         img_w_path = path_to_save + "/images/" + mode +'/'+ str(records["File Name"].values[0])
        
        os.system('cp ' + img_r_path + ' ' + img_w_path)
        
        txt_r_path = path_to_bbox +'/'+img_name+'.txt'
        txt_w_path = path_to_save + '/labels/' + mode
#         txt_w_path = path_to_save + '/labels/' + mode +'/'+ img_name+'.txt'
        if os.path.exists(txt_r_path):
            os.system('cp ' + txt_r_path + ' ' + txt_w_path)
            with open(txt_w_path+'/'+img_name+'.txt', 'r+') as f:
                text = f.read()
                text = re.sub('-1', '0', text)
                f.seek(0)
                f.write(text)
                f.truncate()

'''
    This function is used to group the dataset into different color temperature
    Regular: LL_109-110/NikonD780 + LL_115-116/NikonD780 + LL_125-126/NikonD7000_18mm #214
    Regular (Bright): PAL_FR36_2020-12-13/NikonD780 #55
    Cyan: LL_119-120/NikonD7000_18mm +LL_121-122/NikonD7000_18mm #82
    Blue: PAL_FR13_2020-12-13/* + PAL_FR36_2020-12-13/NikonD7000_18mm + PAL_FR36_2020-12-13/NikonD7000_55mm #104
'''
def split_data_old(qr_df):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='../Complete_SUIT_Dataset', type=str)
    parser.add_argument('--train_ratio', default=0.8, type=float)
    parser.add_argument('--exp_num', default=None, type=str, choices=['exp1', 'exp2', 'exp3', 'exp4'])
    args = parser.parse_args()
    run(args)
    