#!/bin/bash
cd ..
git clone https://github.com/ultralytics/yolov5
cd yolov5 || exit
pip install -r requirements.txt
cd ..
cd YOLO_script || exit

test_image_path='../Datasets/test_images'
path_to_yolov5='../yolov5'
path_to_yolo_model='../yolov5/runs/train/qr_model/weights/best.pt'
path_to_saved_proj_dir="$test_image_path"
path_to_saved_proj_name='yolo_out'

python prepare_yolo_datasets.py
python run_yolo.py --train Y --test_image_path "$test_image_path" --path_to_yolov5 "$path_to_yolov5" --path_to_yolo_model "$path_to_yolo_model" --path_to_saved_proj_dir "$path_to_saved_proj_dir" --path_to_saved_proj_name "$path_to_saved_proj_name"