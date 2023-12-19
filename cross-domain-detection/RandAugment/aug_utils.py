import copy

from PIL.Image import Image
import cv2
import numpy as np
from PIL import Image

def union_of_bboxes(height, width, bboxes, erosion_rate=0.0):
    """Calculate union of bounding boxes.
    ### https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/bbox_utils.py
    Args:
        height (float): Height of image or space.
        width (float): Width of image or space.
        bboxes (List[tuple]): List like bounding boxes. Format is `[(x_min, y_min, x_max, y_max)]`.
        erosion_rate (float): How much each bounding box can be shrinked, useful for erosive cropping.
            Set this in range [0, 1]. 0 will not be erosive at all, 1.0 can make any bbox to lose its volume.

    Returns:
        tuple: A bounding box `(x_min, y_min, x_max, y_max)`.

    """
    x1, y1 = width, height
    x2, y2 = 0, 0
    # make sure it won't influence the bboxes arguments
    bboxes = copy.deepcopy(bboxes)
    bboxes[:, 2] = bboxes[:, 2]+bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]

    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox[:4]
        w, h = x_max - x_min, y_max - y_min
        lim_x1, lim_y1 = x_min + erosion_rate * w, y_min + erosion_rate * h
        lim_x2, lim_y2 = x_max - erosion_rate * w, y_max - erosion_rate * h
        x1, y1 = np.min([x1, lim_x1]), np.min([y1, lim_y1])
        x2, y2 = np.max([x2, lim_x2]), np.max([y2, lim_y2])
    return x1, y1, x2, y2

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
    ## https://blog.paperspace.com/data-augmentation-for-object-detection-rotation-and-shearing/
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
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])

    # compute the new dimensions of the image
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))

    # adjust the rotation matrix with the new dimension
    M[0, 2] += (nw / 2) - cx
    M[1, 2] += (nh / 2) - cy
    return M, nw, nh

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
    corners = (affine_matrix @ corners.T).T

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
    return np.array([x1, y1, x1, y2, x2, y1, x2, y2])

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


def get_distance_matrix(width, height, distortion_scale):
    """Get parameters for ``perspective`` for a random perspective transform.

       Args:
           width (int): width of the image.
           height (int): height of the image.
           distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1.

       Returns:
           List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
           List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
       """
    half_height = height // 2
    half_width = width // 2
    topleft = [
        int(np.random.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
        int(np.random.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()),
    ]
    topright = [
        int(np.random.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
        int(np.random.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()),
    ]
    botright = [
        int(np.random.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
        int(np.random.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item()),
    ]
    botleft = [
        int(np.random.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
        int(np.random.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item()),
    ]
    startpoints = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    endpoints = np.float32([topleft, topright, botright, botleft])
    # return startpoints, endpoints
    M = cv2.getPerspectiveTransform(startpoints, endpoints)
    return M


def get_transformation_matrix(width, height, distortion_scale):
    """Get parameters for ``perspective`` for a random perspective transform.

    Args:
        width (int): width of the image.
        height (int): height of the image.
        distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1.

    Returns:
        List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
        List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
    """
    half_height = height // 2
    half_width = width // 2
    topleft = [
        int(np.random.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
        int(np.random.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()),
    ]
    topright = [
        int(np.random.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
        int(np.random.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()),
    ]
    botright = [
        int(np.random.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
        int(np.random.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item()),
    ]
    botleft = [
        int(np.random.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
        int(np.random.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item()),
    ]
    startpoints = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    endpoints = np.float32([topleft, topright, botright, botleft])
    # return startpoints, endpoints
    
    M = cv2.getPerspectiveTransform(startpoints, endpoints)
    return M