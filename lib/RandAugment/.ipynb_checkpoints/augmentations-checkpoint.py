# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import copy
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image
import cv2
from RandAugment.aug_utils import *
import argparse
from tqdm import tqdm
import os
import glob


def FurtherDistance(img, v, bboxs):  # [0, 1] percentage of original image in t_image
    img = np.rint(np.array(img.im).reshape((img.size[1], img.size[0], 3)))
    o_width, o_height = img.shape[1], img.shape[0]
    width = int(img.shape[1] * v)
    height = int(img.shape[0] * v)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    ## Calculate padding
    delta_w = o_width - width
    delta_h = o_height - height
    left = random.randrange(0, delta_w)
    top = random.randrange(0, delta_h)
    right = delta_w - left
    bottom = delta_h - top

    ## Calcualte new bounding box
    for i in range(len(bboxs)):
        bbox = bboxs[i]
        # print(bbox)
        # print(bbox[0]/o_width, bbox[1]/o_height, bbox[2]/o_width, bbox[3]/o_height)
        x1, y1, w, h = bbox[0]/o_width, bbox[1]/o_height, bbox[2]/o_width, bbox[3]/o_height
        x_start = x1*width + left
        x_end = (x1+w)*width + left
        y_start = y1*height + top
        y_end = (y1+h)*height + top
        bboxs[i][0], bboxs[i][1], bboxs[i][2], bboxs[i][3] = int(x_start), int(y_start), int(x_end-x_start), int(y_end-y_start)

    color = [0, 0, 0]
    t_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    t_img = Image.fromarray(np.rint(t_img).astype(np.uint8))

    return t_img, bboxs


def Perspective(img, v, bbox):  # [0, 1]
    img = np.rint(np.array(img.im).reshape((img.size[1], img.size[0], 3)))
    width, height = img.shape[1], img.shape[0]
    M = get_transformation_matrix(width, height, v)
    # print(np.array(img.im))
    t_img = Image.fromarray(np.rint(cv2.warpPerspective(img, M, (width, height))).astype(np.uint8))
    for i in range(len(bbox)):
        corners = bbox_to_corners(bbox[i]).reshape(1, 4, 2).astype(np.float32)
        updated_corners = np.rint(cv2.perspectiveTransform(corners, M)).reshape(4, 2)
        bbox[i] = corners_to_bbox(updated_corners, t_img)
    return t_img, bbox


def ShearX(img, v, bbox):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    for i in range(len(bbox)):
        corners = bbox_to_corners(bbox[i]).reshape(-1, 2)
        updated_corners = np.vstack((corners[:, 0] - v * corners[:, 1], corners[:, 1])).T.astype(int)
        bbox[i] = corners_to_bbox(updated_corners, img)
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0)), bbox


def ShearY(img, v, bbox):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    for i in range(len(bbox)):
        corners = bbox_to_corners(bbox[i]).reshape(-1, 2)
        updated_corners = np.vstack((corners[:, 0], corners[:, 1] - v * corners[:, 0])).T.astype(int)
        bbox[i] = corners_to_bbox(updated_corners, img)
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0)), bbox


def ShearYBboxSafe(img, v, bbox):
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    # print(v)
    transform_matrix = np.array([[1, 0], [-v, 1]])
    isSafe = True
    orig_box = copy.deepcopy(bbox)
    for i in range(len(bbox)):
        corners = bbox_to_corners(bbox[i]).reshape(-1, 2).T
        corners_t = transform_matrix @ corners
        x_max, x_min = np.max(corners_t[0, :]), np.min(corners_t[0, :])
        y_max, y_min = np.max(corners_t[1, :]), np.min(corners_t[1, :])
        bbox[i][0] = x_min
        bbox[i][1] = y_min
        bbox[i][2] = x_max - x_min
        bbox[i][3] = y_max - y_min
        if x_min < 0 or y_min < 0 or x_max > img.size[0] or y_max > img.size[1]:
            isSafe = False
    if isSafe:
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0)), bbox
    else:
        return img, orig_box


def ShearXBboxSafe(img, v, bbox):
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    # print(v)
    transform_matrix = np.array([[1, -v], [0, 1]])
    for i in range(len(bbox)):
        corners = bbox_to_corners(bbox[i]).reshape(-1, 2).T
        corners_t = transform_matrix @ corners
        x_max, x_min = np.max(corners_t[0, :]), np.min(corners_t[0, :])
        y_max, y_min = np.max(corners_t[1, :]), np.min(corners_t[1, :])
        bbox[i][0] = x_min
        bbox[i][1] = y_min
        bbox[i][2] = x_max - x_min
        bbox[i][3] = y_max - y_min
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0)), bbox


def TranslateX(img, v, bbox):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0)), bbox


def TranslateXabs(img, v, bbox):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    for i in range(len(bbox)):
        corners = bbox_to_corners(bbox[i]).reshape(-1, 2)
        updated_corners = np.vstack((corners[:, 0] - v, corners[:, 1])).T.astype(int)
        bbox[i] = corners_to_bbox(updated_corners, img)
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0)), bbox


def TranslateXBBoxSafe(img, v, bbox):  # [0, 1]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v

    max_dist = img.size[1]
    for box in bbox:
        if v < 0:
            max_dist = min(max_dist, img.size[0] - box[0] - box[2])
        else:
            max_dist = min(max_dist, box[0])

    v = float(v * max_dist)
    for i in range(len(bbox)):
        corners = bbox_to_corners(bbox[i]).reshape(-1, 2)
        updated_corners = np.vstack((corners[:, 0] - v, corners[:, 1])).T.astype(int)
        bbox[i] = corners_to_bbox(updated_corners, img)
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0)), bbox


def TranslateY(img, v, bbox):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v)), bbox


def TranslateYabs(img, v, bbox):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    for i in range(len(bbox)):
        corners = bbox_to_corners(bbox[i]).reshape(-1, 2)
        updated_corners = np.vstack((corners[:, 0], corners[:, 1] - v)).T.astype(int)
        bbox[i] = corners_to_bbox(updated_corners, img)
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v)), bbox


def TranslateYBboxSafe(img, v, bbox):  # [0, 1]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v

    max_dist = img.size[0]
    for box in bbox:
        if v < 0:
            max_dist = min(max_dist, img.size[1] - box[1] - box[3])
        else:
            max_dist = min(max_dist, box[1])

    v = float(v * max_dist)
    for i in range(len(bbox)):
        corners = bbox_to_corners(bbox[i]).reshape(-1, 2)
        updated_corners = np.vstack((corners[:, 0], corners[:, 1] - v)).T.astype(int)
        bbox[i] = corners_to_bbox(updated_corners, img)
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v)), bbox


def Rotate(img, v, bbox):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    M = rotation_matrix(img, v)
    for i in range(len(bbox)):
        bbox[i] = rotate_update_bbox(bbox[i], M, img)
    return img.rotate(v), bbox


def AutoContrast(img, _, bbox):
    return PIL.ImageOps.autocontrast(img), bbox


def Invert(img, _, bbox):
    return PIL.ImageOps.invert(img), bbox


def Equalize(img, _, bbox):
    return PIL.ImageOps.equalize(img), bbox


def Flip(img, _, bbox):  # not from the paper
    return PIL.ImageOps.mirror(img), bbox


def Solarize(img, v, bbox):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v), bbox


def SolarizeAdd(img, addition=0, bbox=None, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold), bbox


def Posterize(img, v, bbox):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v), bbox


def Contrast(img, v, bbox):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v), bbox


def Color(img, v, bbox):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v), bbox


def Brightness(img, v, bbox):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v), bbox


def Sharpness(img, v, bbox):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v), bbox


def Cutout(img, v, bbox):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v, bbox)


def CutoutAbs(img, v, bbox):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img, bbox


def CutoutBboxSafe(img, v, bbox):  # [0, 1] => [0, 1 * the precentage of maximum cutout without cutting out bbox]
    # assert 0 <= v <= 20
    if v < 0:
        return img

    img_w, img_h = img.size

    while True:
        img_x0 = np.random.uniform(img_w)
        img_y0 = np.random.uniform(img_h)
        img_x0 = int(max(0, img_x0 - v / 2.))
        img_y0 = int(max(0, img_y0 - v / 2.))
        img_x1 = min(img_w, img_x0 + v)
        img_y1 = min(img_h, img_y0 + v)
        isValid = True
        for box in bbox:
            if (img_x0 <= box[0] and img_x1 >= box[0]) or \
                    (img_x0 <= box[0] + box[2] and img_x1 >= box[0] + box[2]) or \
                    (img_x0 >= box[0] and img_x1 <= box[0] + box[2]) or \
                    (img_y0 <= box[1] and img_y1 >= box[1]) or \
                    (img_y0 <= box[1] + box[3] and img_y1 >= box[1] + box[3]) or \
                    (img_y0 >= box[1] and img_y1 <= box[1] + box[3]):
                isValid = False
                break
        if isValid:
            break

    xy = (img_x0, img_y0, img_x1, img_y1)
    r, g, b = img.getchannel('R'), img.getchannel('G'), img.getchannel('B')
    color = (int(np.mean(r)), int(np.mean(g)), int(np.mean(b)))
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img, bbox


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, v, bbox):
    return img, bbox


augment_map = {
    'AutoContrast': AutoContrast,
    'Brightness': Brightness,
    'Contrast': Contrast,
    'Color': Color,
    'Cutout': Cutout,
    'CutoutAbs': CutoutAbs,
    'Equalize': Equalize,
    'Identity': Identity,
    'Invert': Invert,
    'Posterize': Posterize,
    'Rotate': Rotate,
    'Sharpness': Sharpness,
    'ShearX': ShearX,
    'ShearY': ShearY,
    'Solarize': Solarize,
    'SoloarizeAdd': SolarizeAdd,
    'TranslateX': TranslateX,
    'TranslateY': TranslateY,
    'TranslateXabs': TranslateXabs,
    'TranslateYabs': TranslateYabs,
    'TranslateXthr': TranslateXBBoxSafe,
    'TranslateYthr': TranslateYBboxSafe,
    'CutoutBboxSafe': CutoutBboxSafe,
    'FurtherDistance': FurtherDistance
}

aug_sel = [
    'AutoContrast', 'Equalize', 'Rotate', 'Posterize', 'Color', 'Contrast',
    'Brightness', 'Sharpness', 'ShearX', 'ShearY', 'CutoutAbs', 'TranslateXabs', 'FurtherDistance',
    'TranslateYabs', 'Cutout', 'TranslateX', 'TranslateY', 'TranslateXthr', 'TranslateYthr', 'CutoutBboxSafe'
]


def augment_list(aug_ls=None):  # 16 oeprations and their ranges

    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    # l = [
    #     (Identity, 0., 1.0),
    #     (ShearX, 0., 0.3),  # 0
    #     (ShearY, 0., 0.3),  # 1
    #     (TranslateX, 0., 0.33),  # 2
    #     (TranslateY, 0., 0.33),  # 3
    #     (Rotate, 0, 30),  # 4
    #     (AutoContrast, 0, 1),  # 5
    #     (Invert, 0, 1),  # 6
    #     (Equalize, 0, 1),  # 7
    #     (Solarize, 0, 110),  # 8
    #     (Posterize, 4, 8),  # 9
    #     # (Contrast, 0.1, 1.9),  # 10
    #     (Color, 0.1, 1.9),  # 11
    #     (Brightness, 0.1, 1.9),  # 12
    #     (Sharpness, 0.1, 1.9),  # 13
    #     # (Cutout, 0, 0.2),  # 14
    #     # (SamplePairing(imgs), 0, 0.4),  # 15
    # ]

    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    # l = [
    #     (AutoContrast, 0, 1),
    #     (Equalize, 0, 1),
    #     # (Invert, 0, 1),
    #     (Rotate, 0, 30),
    #     (Posterize, 0, 4),
    #     # (Solarize, 0, 256),
    #     # (SolarizeAdd, 0, 110),
    #     (Color, 0.1, 1.9),
    #     (Contrast, 0.1, 1.9),
    #     (Brightness, 0.1, 1.9),
    #     (Sharpness, 0.1, 1.9),
    #     (ShearX, 0., 0.3),
    #     (ShearY, 0., 0.3),
    #     (CutoutAbs, 0, 40),
    #     (TranslateXabs, 0., 100),
    #     (TranslateYabs, 0., 100),
    # ]

    if aug_ls is None:
        ls = [
            # (AutoContrast, 0, 1),
            # (Equalize, 0, 1),
            # # (Invert, 0, 1),
            # (Rotate, 0, 30),
            # (Posterize, 0, 4),
            # # (Solarize, 0, 256),
            # # (SolarizeAdd, 0, 110),
            # (Color, 0.1, 1.9),
            # (Contrast, 0.1, 1.9),
            # (Brightness, 0.1, 1.9),
            # (Sharpness, 0.1, 1.9),
            # (TranslateXBBoxSafe, 0., 1),
            # (TranslateYBboxSafe, 0., 1),
            # (CutoutBboxSafe, 0, 100),
            # (ShearXBboxSafe, 0., 0.3),
            # (ShearYBboxSafe, 0., 0.3),
#             (Perspective, 0, 0.5)
            (FurtherDistance, 0, 1)
        ]
    else:
        ls = []
        for aug, minval, maxval in aug_ls:
            ls.append((augment_map[aug_sel[aug]], minval, maxval))
    return ls


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class RandAugment:
    def __init__(self, n, m, _augment_list=None):
        self.n = n
        self.m = m  # [0, 30]
        self.augment_list = augment_list(_augment_list)
        # self.ops = None

    def __call__(self, img, bbox):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval

            # change x1, y1, x2, y2 to x1, y1, w, h
            bbox[:,2] = bbox[:,2] - bbox[:,0]
            bbox[:,3] = bbox[:,3] - bbox[:,1]

            img, bbox = op(img, val, bbox)

            # change x1, y1, w, h to x1, y1, x2, y2
            bbox[:,2] = bbox[:,2] + bbox[:,0]
            bbox[:,3] = bbox[:,3] + bbox[:,1]

        return img, bbox

    def test(self, img, bbox):
        ops = [
            (SolarizeAdd, 0, 110),
            (Rotate, 0, 30),
            (ShearX, 0., 0.3),
            (ShearY, 0., 0.3),
            (TranslateXabs, 0., 100),
            (TranslateYabs, 0., 100)
        ]
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img, bbox = op(img, val, bbox)
        return img, bbox


def augment(data_path, n, m):
    """
    Random Augment a whole directory
    """
    img_dirs = glob.glob(f"{data_path}/**/*.JPG", recursive=True)
    print("RandAugmenting...")
    for img_dir in tqdm(img_dirs):
        img = cv2.imread(img_dir)
        im = Image.fromarray(img)
        label_dir = img_dir.replace("images", "labels").replace("JPG", "txt")
        if os.path.exists(label_dir):
            randaugment = RandAugment(n, m)
            with open(label_dir, "r+") as f:
                lines = list(set(f.readlines()))
                bbox = []
                for line in lines:
                    line = line.split(" ")
                    line[1] = int(float(line[1]) * img.shape[1])
                    line[2] = int(float(line[2]) * img.shape[0])
                    line[3] = int(float(line[3]) * img.shape[1])
                    line[4] = int(float(line[4]) * img.shape[0])
                    bbox.append([int(line[1]), int(line[2]), int(line[3]), int(line[4]), ])
                im, bbox = randaugment(im, bbox)
                cv2.imwrite(img_dir, np.array(im))
                res = []
                for i in range(len(bbox)):
                    line = lines[i].split(" ")
                    line[1] = str(float(line[1]) / img.shape[1])
                    line[2] = str(float(line[2]) / img.shape[0])
                    line[3] = str(float(line[3]) / img.shape[1])
                    line[4] = str(float(line[4]) / img.shape[0])
                    res.append(" ".join(line))
                f.truncate(0)
                f.write("".join(res))


def main(args):
    """
    The main function that conducts rand augmentation
    """
    if not os.path.exists(args.out_path):
        print("initializing the directory")
        os.system(f"mkdir {args.out_path}")
        os.system(f"mkdir {args.out_path}/images")
        os.system(f"mkdir {args.out_path}/labels")
        os.system(f"cp -r {args.dataset_path}/images {args.out_path}")
        os.system(f"cp -r {args.dataset_path}/labels {args.out_path}")
    augment(args.out_path, args.n, args.m)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="../Complete_SUIT_Dataset_corrected", type=str,
                        help="path to the dataset")
    parser.add_argument("--out_path", type=str, help="path to the output directory")
    parser.add_argument("--n", default=5, type=int, help="the mean of transformation after correction")
    parser.add_argument("--m", default=2, type=int, help="the standard deviation of transformation after correction")
    args = parser.parse_args()
    main(args)
