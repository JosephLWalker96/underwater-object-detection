import argparse
import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch import ToTensorV2
import albumentations as A
from tqdm import tqdm

from model import net

'''
    To run this file, the images directory should look like:
        img_dir/
            img1.JPG
            img2.PNG
            img3.JPG
            ...
'''


class DetectDataset(Dataset):
    def __init__(self):
        self.num_class = 3  # SUIT + target + background
        self.transform = A.Compose([
            A.Resize(512, 512),
            ToTensorV2(p=1.0)
        ])
        self.img_paths = []
        self.img_names = []
        for filename in os.listdir(args.path_to_images):
            img_name, file_ext = os.path.splitext(filename)
            if file_ext == ".JPG" or file_ext == '.PNG':
                self.img_paths.append(os.path.join(args.path_to_images, filename))
                self.img_names.append(img_name)

    def __getitem__(self, item):
        img_path = self.img_paths[item]
        img_name = self.img_names[item]
        img = np.array(cv2.imread(img_path))/255.0
        sample = self.transform(image=img.astype(np.float32))
        img = sample['image']
        return img, img_name, img_path

    def __len__(self):
        return len(self.img_paths)


def save_txt(img_name, box, label):
    if not os.path.exists(args.path_to_output + '/labels'):
        os.mkdir(args.path_to_output + '/labels')
    with open(args.path_to_output + '/labels/' + img_name + '.txt', 'a') as f:
        xs = float((box[0] + box[2]) / 2)
        ys = float((box[1] + box[3]) / 2)
        w = float(box[2] - box[0])
        h = float(box[3] - box[1])
        line = str(int(label)) + ' ' + str(float(xs)) + ' ' + str(float(ys)) + ' ' + str(float(w)) + ' ' + str(
            float(h)) + '\n'
        f.write(line)
    f.close()


def draw_bbox(img, box):
    x1 = int(box[0] * img.shape[1])
    y1 = int(box[1] * img.shape[0])
    x2 = int(box[2] * img.shape[1])
    y2 = int(box[3] * img.shape[0])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 10)


def detect(dataset: Dataset, model: net):
    dataloader = DataLoader(dataset, pin_memory=True, num_workers=4, batch_size=args.batch_size)
    rslt_df = pd.DataFrame(columns=['image_name', 'x1', 'y1', 'x2', 'y2', 'label'])
    for images, img_names, img_paths in tqdm(dataloader):
        outputs = model(images)
        for idx in range(len(outputs)):
            boxes = outputs[idx]['boxes']/512.0
            labels = outputs[idx]['labels']
            img_name = img_names[idx]
            img_path = img_paths[idx]
            img = cv2.imread(img_path)

            for box, label in zip(boxes, labels):
                rslt_df = rslt_df.append({
                    'image_name': img_name,
                    'x1': int(box[0]),
                    'y1': int(box[1]),
                    'x2': int(box[2]),
                    'y2': int(box[3]),
                    'label': int(label)
                }, ignore_index=True)

                if args.save_txt:
                    save_txt(img_name, box, label)

                if args.save_image:
                    draw_bbox(img, box)

            if args.save_image:
                if not os.path.exists(args.path_to_output + '/image_with_bbox'):
                    os.mkdir(args.path_to_output + '/image_with_bbox')
                cv2.imwrite(args.path_to_output + '/image_with_bbox/' + img_name + '.JPG', img)
    rslt_df.to_csv(os.path.join(args.path_to_output, 'detect.csv'))

def main():
    with open(args.path_to_model, 'rb') as f:
        model = torch.load(args.path_to_model, map_location=device)
    dataset = DetectDataset()
    detect(dataset, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_model', default='../Complete_SUIT_Dataset/models/faster-rcnn', type=str)
    parser.add_argument('--path_to_images', type=str)
    parser.add_argument('--path_to_output', default='../detect', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--save_image', default=False, action='store_true')
    parser.add_argument('--save_txt', default=False, action='store_true')
    parser.add_argument('--save_csv', default=True, action='store_true')
    args = parser.parse_args()
    assert os.path.exists(args.path_to_model)
    assert os.path.exists(args.path_to_images)
    assert args.path_to_output is not None
    if not os.path.exists(args.path_to_output):
        os.mkdir(args.path_to_output)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    main()
