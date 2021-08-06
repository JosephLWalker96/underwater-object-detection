import os
from tqdm import tqdm
import argparse


def detect(path_to_yolov5, abs_path_to_yolo_model, abs_path_to_test_images, saved_proj_dir, saved_proj_name):
    current_path = os.getcwd()
    os.chdir(path_to_yolov5)

    if os.path.exists(saved_proj_dir+"/"+saved_proj_name):
        os.system("rm -r "+saved_proj_dir+"/"+saved_proj_name)

    # need to declare which images to test on later
    test_cmd = 'python detect.py' + ' --img 640 ' + '--weights ' + abs_path_to_yolo_model + ' --save-txt'
    test_cmd = test_cmd + ' --project ' + saved_proj_dir + ' --name ' + saved_proj_name + ' --line-thickness 10 ' + ' --exist-ok'

    directory = abs_path_to_test_images
    # for loc_name in tqdm(os.listdir(directory)):
    #     loc_path = directory + "/" + loc_name
    #     if os.path.isdir(loc_path):
    #         for cam_name in os.listdir(loc_path):
    #             cam_path = loc_path + "/" + cam_name
    #             if os.path.isdir(cam_path):
    #                 # for filename in os.listdir(cam_path):
    #                 #     img_name, img_ext = os.path.splitext(filename)
    #                 #     if img_ext == ".JPG" or img_ext == ".PNG":
    #                 #         img_path = cam_path + "/" + filename
    #                         # run the detection command
    #                 os.system(test_cmd + ' --source ' + cam_path)
    os.system(test_cmd + ' --source ' + directory)

    os.chdir(current_path)


def train(path_to_yolov5, path_to_yolo_model):
    current_path = os.getcwd()
    os.chdir(path_to_yolov5)
    if os.path.exists("../yolov5/runs/train/qr_model"):
        os.system("rm -r ..yolov5/runs/train/qr_model")
    # yolo_model = path_to_yolo_model.split('/')
    train_cmd = 'python train.py --img 640 --batch 4 --epochs 15 --workers 4 --data qr_code.yaml --weights yolov5s.pt'
    train_cmd = train_cmd + ' --project runs/train --name qr_model'
    os.system(train_cmd)
    os.chdir(current_path)


def run(args):
    if args.train == 'Y':
        train(args.path_to_yolov5, args.path_to_yolo_model)

    detect(args.path_to_yolov5, args.path_to_yolo_model, args.test_image_path,
           args.path_to_saved_proj_dir, args.path_to_saved_proj_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='N', type=str)
    parser.add_argument('--test_image_path', type=str)
    parser.add_argument('--path_to_yolov5', type=str)
    parser.add_argument('--path_to_yolo_model', type=str)
    parser.add_argument('--path_to_saved_proj_dir', type=str)
    parser.add_argument('--path_to_saved_proj_name', type=str)
    args = parser.parse_args()
    run(args)
