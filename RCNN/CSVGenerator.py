import argparse
import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm

def run(args):
    path_to_images = args.image_path
    path_to_bbox = args.label_path
    
    # check whether the path exists
    if path_to_images is None:
        print("Please specify image path with '--image_path path_to_images'")
        raise FileNotFoundError

    if path_to_bbox is None:
        print("Please specify label path with '--label_path path_to_bbox_label'")
        raise FileNotFoundError

    if not os.path.exists(path_to_images):
        print("Label path does not exist")
        raise FileNotFoundError

    if not os.path.exists(path_to_bbox):
        print("Label path does not exist")
        raise FileNotFoundError
    
    directory = path_to_images
    qr_df = pd.DataFrame(columns = ["Location", "Camera", "File Name", "Image", "Image Width", "Image Height"])
  
    print("Getting Images Info")
    for loc_name in tqdm(os.listdir(directory)):
        loc_path = directory+"/"+loc_name
        if os.path.isdir(loc_path):
            for cam_name in os.listdir(loc_path):
                cam_path = loc_path+"/"+cam_name
                if os.path.isdir(cam_path):
                    for filename in os.listdir(cam_path):
                        img_name, img_ext = os.path.splitext(filename)
                        if img_ext == ".JPG" or img_ext == ".PNG":
                            img_path = cam_path+"/"+filename
                            img = np.array(cv2.imread(img_path))
                            qr_df = qr_df.append({
                                "Location":loc_name,
                                "Camera":cam_name,
                                "File Name":filename,
                                "Image":img_name,
                                "Image Width":img.shape[1],
                                "Image Height":img.shape[0]
                            }, ignore_index = True)
    
    # this following may need to be modifed
    qr_df = qr_df.drop_duplicates(subset=['Image'], keep='first')
    
    # The following is adding bbox data
    qr_df['x'] = -1
    qr_df['y'] = -1
    qr_df['w'] = -1
    qr_df['h'] = -1
    
    print("Getting SUIT Info")
    directory = path_to_bbox
    for bboxes_txt in tqdm(os.listdir(directory)):
        txt_path = directory+"/"+bboxes_txt
        img_name = os.path.splitext(bboxes_txt)[0]
        
        # this image's bbox have two lines of data
        if img_name == 'DSC_6775':
            continue

        if qr_df["Image"][qr_df["Image"] == img_name].count() == 0:
            continue

        coord = np.loadtxt(txt_path)

        records = qr_df.loc[qr_df["Image"]==img_name]

        width = int(records['Image Width'])
        height = int(records['Image Height'])

        xs = int(coord[1] * width)
        ys = int(coord[2] * height)

        w = int(coord[3] * width)
        h = int(coord[4] * height)

        x1 = xs -int(.5*w)
        x2 = xs +int(.5*w)

        y1= ys -int(.5*h)
        y2 = ys+int(.5*h)

        qr_df.loc[qr_df["Image"]==img_name, 'x'] = x1
        qr_df.loc[qr_df["Image"]==img_name, 'y'] = y1
        qr_df.loc[qr_df["Image"]==img_name, 'w'] = w
        qr_df.loc[qr_df["Image"]==img_name, 'h'] = h
                            
    train_qrdf = qr_df[qr_df["x"]!=-1]
    all_qrdf = qr_df
    
    train_qrdf.to_csv(path_to_images+"/train_qr_labels.csv")
    print("Saved training csv to " + path_to_images+"/train_qr_labels.csv")
#     all_qrdf.to_csv(path_to_images+"/qr_labels.csv")
#     print("Saved all csv to " + path_to_images+"/qr_labels.csv")
                            
                            
                            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--label_path', type=str)
    args = parser.parse_args()
    run(args)