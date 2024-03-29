from PIL.Image import Image
import cv2
import numpy as np
from PIL import Image

def rotate_im(img, bbox, angle):
    """
    ### https://github.com/Paperspace/DataAugmentationForObjectDetection/blob/df227363405a46c889255a0628fa5d84439c2e03/data_aug/bbox_util.py#L180
    rotate an image counterclockwise with the value of angle. The image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored black
    :img: PIL image
        the img to be rotated
    :bbox: np.array of 4 integers [x1, y1, w, h]
        the bbox of the image
    :angle: float
        the degree to be rotated. Positive means counterclockwise, and vice versa
    :return: tuple of a PIL image and the updated bbox (updated_img, [nx1, ny1, nw, nh])
    """
    # compute the rotation matrix
    M, (nw, nh) = rotation_matrix(img, angle)

    # perform the actual rotation and return the image
    img_np = np.array(img).astype(np.int)
    img_np = img_np.astype(np.uint8)
    img = cv2.warpAffine(img_np, M, (nw, nh))

    # compute the new bounding box
    bbox = rotate_update_bbox(bbox, M, img)

    return Image.fromarray(img)

def rotation_matrix(img, angle):
    """
    get the rotation matrix of given angle and img
    :img: PIL image
        the img to be rotated
    :angle: float
        the degree of the image to be rotated. Positive means counterclockwise, and vice versa
    :return: the np.ndarray of affine matrix of rotation, with the occupied pixels of the original image blackened
    """
    # dimension and center of the image
    (w, h) = img.size
    (cx, cy) = (w // 2, h // 2)

    # compute the rotation matrix
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    # cos = np.abs(M[0,0])
    # sin = np.abs(M[0,1])

    # compute the new dimensions of the image
    # nw = int((h * sin) + (w * cos))
    # nh = int((h * cos) + (w * sin))

    # adjust the rotation matrix with the new dimension
    # M[0, 2] += (nw / 2) - cx
    # M[1, 2] += (nh / 2) - cy
    return M #, (nw, nh)

def rotate_update_bbox(bbox, affine_matrix, img):
    """
    This function is called after updated affine matrix is computed,
    should produce the updated bbox of the image
    :bbox: np.array of 4 integers
        [x1, y1, width, height]
    :affine_matrix: 2 x 3 np.ndarray of np.float64 
        the affine matrix computed
    :img: the PIL image to be transformed
    :return: np.array of 4 integers
        the updated bounding box of the image [x1`,y1`,width`,height`]
    """
    # get the corners of the bbox
    corners = bbox_to_corners(bbox)

    # reshape the corners to N x 2, where N is 4, the number of coordinates of corners
    corners = corners.reshape(-1, 2)
    # In order to do affine transformation, insert 1 at the end of each row
    corners = np.insert(corners, 2, 1, axis = 1)

    # first update the corners using the affine transformation
    corners = (affine_matrix @ corners.T).T.astype(int)

    # updated bbox set bounding box to be within the bounding dimension
    bbox = corners_to_bbox(corners, img)
 
    return bbox

def bbox_to_corners(bbox):
    """
    the function takes in the default bbox and return the 4 corners of the image
    x1,y1 -- x2,y1
    |          |
    |          |
    |          |
    x1,y2 -- x2,y2
    :bbox: np.array of 4 integers
        [x1, y1, width, height]
    :return: np.array of 4 integers
        [x1, y1, x2, y1, x1, y2, x2, y2]
    """
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[0] + bbox[2]
    y2 = bbox[1] + bbox[3]
    return np.array([x1, y1, x2, y1, x1, y2, x2, y2])

def corners_to_bbox(corners, img):
    """
    Convert the coordinates of 4 corners to the bbox, [x1, y1, w, h]
    :corners: np.ndarray of 4 x 2 integers
        the array of corners
    :img: PIL image that is transformed
    :return: np.array of 4 integers
        bounding boxes
    """
    w,h = img.size
    x1 = max(0, min(corners[:,0]))
    y1 = max(0, min(corners[:,1]))
    width = min(w, max(corners[:,0])) - x1
    height = min(h, max(corners[:,1])) - y1
    return np.array([x1, y1, width, height])