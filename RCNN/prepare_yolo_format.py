import pandas as pd
import os
import re
import tqdm

def main():
    train_csv = '../Datasets/train_images/train_qr_labels.csv'
    test_csv = '../Datasets/test_images/test_qr_labels.csv'

    path_to_load_train_images = '../Datasets/train_images'
    path_to_load_test_images = '../Datasets/test_images'

    if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
        os.system('python split_data.py')

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # splitting training and validation data
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    val_df = train_df[:int(train_df.__len__() * 0.2)]
    train_df = train_df[int(train_df.__len__() * 0.2):]
    train_df.index = pd.RangeIndex(len(train_df.index))
    val_df.index = pd.RangeIndex(len(val_df.index))

    # remove if exists
    # if not os.path.exists('../Datasets/YOLO'):
#         os.system('rm -r ../Datasets/YOLO')

    # create the place to store
    path_to_yolo = '../Datasets'
    path_to_save_images = path_to_yolo + '/' + 'images'
    path_to_save_labels = path_to_yolo + '/' + 'labels'
    os.system('mkdir ' + path_to_yolo)
    os.system('mkdir ' + path_to_save_images)
    os.system('mkdir ' + path_to_save_labels)

    # handling train
    os.system('mkdir ' + path_to_save_images + '/train')
    os.system('mkdir ' + path_to_save_labels + '/train')
    save_to_path(train_df, path_to_load_train_images, path_to_save_images + '/train', path_to_save_labels + '/train')

    # handling val
    os.system('mkdir ' + path_to_save_images + '/val')
    os.system('mkdir ' + path_to_save_labels + '/val')
    save_to_path(val_df, path_to_load_train_images, path_to_save_images + '/val', path_to_save_labels + '/val')

    # handling test
    os.system('mkdir ' + path_to_save_images + '/test')
    os.system('mkdir ' + path_to_save_labels + '/test')
    save_to_path(test_df, path_to_load_test_images, path_to_save_images + '/test', path_to_save_labels + '/test')


def save_to_path(df, path_to_load_images, path_to_save_images, path_to_save_labels, csv_filename=None):
    if csv_filename:
        df.to_csv(path_to_save_images + '/' + csv_filename)
    for idx in tqdm.tqdm(range(df.__len__())):
        records = df[df.index == idx]
        img_r_path = path_to_load_images + "/" + str(records["Location"].values[0]) + "/" + \
                     str(records["Camera"].values[0]) + "/" + str(records["File Name"].values[0])
        txt_r_path = '../Datasets/bboxes_for_SUIT_images_2021-05-10-02-29-37/' + str(
            records['Image'].values[0]) + '.txt'
        img_w_path = path_to_save_images + "/" + str(records["File Name"].values[0])
        txt_w_path = path_to_save_labels + "/" + str(records['Image'].values[0]) + '.txt'
        os.system('mv ' + img_r_path + ' ' + img_w_path)

        if os.path.exists(txt_r_path):
            os.system('mv ' + txt_r_path + ' ' + txt_w_path)

            with open(txt_w_path, 'r+') as f:
                text = f.read()
                text = re.sub('-1', '0', text)
                f.seek(0)
                f.write(text)
                f.truncate()


if __name__ == '__main__':
    main()
