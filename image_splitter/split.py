import numpy as np
import cv2
import pandas as pd

ANNOTATION = pd.read_csv("../Datasets/labels.csv", header = None)

def split_image(im_path, window_size_0=100, window_size_1=100, margin=0, overlap=0):
    """
    Wrapper function that calls split and generate_etch_image functions
    :im_path: the path of the original image
    :window_size_0: the height of the splitted image
    :window_size_1: the width of the splitted image
    :margin: the margin on the sides of the original image when splitted
    :overlap: the proportion of the images overlapping over the ones next to it
    :return: the splitted image and the new splitted ground truth, in the format of
    ([splitted_img1, splitted_img2, ...], [(index of subimage1 containing new bounding box, subimage1 bbox), 
                                            (index of subimage2 containing new bounding box, subimage2 bbox), ...])
    """
    img = cv2.imread(im_path)
    splitted = split(img, window_size_0, window_size_1, margin, overlap)
    splitted_truth = generate_etch_image(im_path, window_size_0, window_size_1, margin, overlap)
#     print("The splitted ground truth is", splitted_truth)
    return (splitted, splitted_truth)

def split(img, window_size_0, window_size_1, margin, overlap):
    """
    Split function that takes in an image and split it with the specified pixel size, 
    margin, and overlapping portion
    :params:
    :img: the img to be splitted
    :window_size_0: the height of the splitted image
    :window_size_1: the width of the splitted image
    :margin: the margin on the sides of the original image when splitted
    :overlap: the proportion of the images overlapping over the ones next to it
    :return: the splitted image
    """
    # get the shape of the original image
    sh = list(img.shape)
    # compute the shape with user specified margin
    sh[0], sh[1] = sh[0] + margin * 2, sh[1] + margin * 2
    # specify the empty image with newly computed shape
    img_ = np.zeros(shape=sh)
    img_[margin:-margin, margin:-margin] = img
    
    # compute the strides
    stride_0 = int(window_size_0 * (1-overlap))
    step_0 = window_size_0 + 2 * margin
    
    stride_1 = int(window_size_1 * (1-overlap))
    step_1 = window_size_1 + 2 * margin
    
    # compute the number of images on the rows and columns
    nrows, ncols = int(img.shape[0] // (window_size_0 * (1 - overlap))), int(img.shape[1] // (window_size_1 * (1 - overlap)))

    splitted = []
#     fig = plt.figure(figsize=(10, 7))
    counter = 1
    for i in range(nrows):
        for j in range(ncols):
            # compute the next start
            h_start = j*stride_1
            v_start = i*stride_0
            
            # determine if the sub image reaches the end of the main image
            if v_start + step_0 > sh[0]:
                v_start = sh[0] - step_0
            if h_start + step_1 > sh[0]:
                h_start = sh[1] - step_1
            # crop the image
            cropped = img_[v_start:v_start+step_0, h_start:h_start+step_1]
            splitted.append(cropped)
#             fig.add_subplot(nrows, ncols, counter)
#             plt.imshow(cropped/255)
            counter += 1
    return splitted


def generate_etch_image(im_path, window_size_0, window_size_1, margin, overlap):
    """
    The function takes in an image and retrieve the annotated information of the image and create a new blank image.
    Using the annotated bounding box, detect the ground truth in the splitted images
    :param:
    :im_path: the path of the image to generate the etching image of
    :out_path: the directory of the image to generate of the etching image into
    """
    # reading the image
    img = cv2.imread(im_path)
    # get the shape of the original image
    sh = list(img.shape)
    
    # retrieve the bounding box annotations
    row = ANNOTATION[ANNOTATION[5] == im_path.split("/")[-1]]
    # if the length of the row is zero, the image is not annotated, return None
    if len(row) == 0:
        return None
    info = row.values[0]

    # initialize an etching black image with width and height in the info row
    etch = np.zeros((info[7], info[6], 3))
    # fill the bounding box of the image with red
    etch = cv2.rectangle(etch, (info[1], info[2]), (info[1]+info[3], info[2]+info[4]), (255,0,0), -1)
    
    # implement the split function on the etched image
    etch_splitted = split(etch, window_size_0, window_size_1, margin, overlap)
    
    splitted_truths = []
    for ii,im in enumerate(etch_splitted):
        splitted_truth = detect_splitted_truth(im)
        if splitted_truth:
            splitted_truths.append((ii, splitted_truth))
    return splitted_truths

def detect_splitted_truth(img):
    """
    The function to detect the red portion in the splitted images and return the bounding box information 
    """
    # get the x and y coordinates of the ground truth in the splitted image
    X,Y = np.where(np.all(img==(255,0,0),axis=2))
    if len(X) != 0 and len(Y) != 0:
        # if the ground truth is detected in the subimage, return the new coordinates of bounding box
        bbox = (min(Y), min(X), max(Y)-min(Y), max(X)-min(X))
        return bbox
    else:
        # else, return None
        return None