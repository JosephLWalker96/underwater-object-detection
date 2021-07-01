import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm

def run(path_to_images = None):
    if path_to_images:
        # Input for Images Folder
        print("Please Enter The Path To Images Folder For Training")
        path_to_images = input()
        print("You Just Enter: " + path_to_images)

        # check whether the path exists
        while not os.path.exists(path_to_images):
            print("Path does not exist")
            print("Please Re-Enter The Path To Images For Training")
            path_to_images = input()
            print("You Just Enter: " + path_to_images)
    
    # Inputs for Text Files Folder
    print("Please Enter The Path To Folder of Text Files With SUIT Location")
    path_to_bbox = input()
    print("You Just Enter: " + path_to_bbox)
    
    # check whether the path exists
    while not os.path.exists(path_to_bbox):
        print("Path does not exist")
        print("Please Re-Enter The Path To Folder of Text Files With SUIT Location")
        path_to_bbox = input()
        print("You Just Enter: " + path_to_bbox)
    
    directory = path_to_images
    qr_df = pd.DataFrame(columns = ["id", "Location", "Camera", "File Name", "Image", "Image Width", "Image Height"])
  
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
    #                             "id":idx,
                                "Location":loc_name,
                                "Camera":cam_name,
                                "File Name":filename,
                                "Image":img_name,
                                "Image Width":img.shape[0],
                                "Image Height":img.shape[1]
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

        coord = np.loadtxt(txt_path)

        records = qr_df.loc[qr_df["Image"]==img_name]

        image = cv2.imread(img_path)
        hor_len = int(records['Image Height'])
        vert_len = int(records['Image Width'])

        xs = int(coord[1] * hor_len)
        ys = int(coord[2] * vert_len)

        h = int(coord[3] * hor_len)
        w = int(coord[4] * vert_len)

        x1 = xs -int(.5*h)
        x2 = xs +int(.5*h)

        y1= ys -int(.5*w)
        y2 = ys+int(.5*w)

        qr_df.loc[qr_df["Image"]==img_name, 'x'] = x1
        qr_df.loc[qr_df["Image"]==img_name, 'y'] = y1
        qr_df.loc[qr_df["Image"]==img_name, 'w'] = h
        qr_df.loc[qr_df["Image"]==img_name, 'h'] = w
                            
    train_qrdf = qr_df[qr_df["x"]!=-1]
    all_qrdf = qr_df
    
    train_qrdf.to_csv(path_to_images+"/train_qr_labels.csv")
    print("Saved training csv to " + path_to_images+"/train_qr_labels.csv")
#     all_qrdf.to_csv(path_to_images+"/qr_labels.csv")
#     print("Saved all csv to " + path_to_images+"/qr_labels.csv")
                            
                            
                            
if __name__ == "__main__":
    run()                        
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            