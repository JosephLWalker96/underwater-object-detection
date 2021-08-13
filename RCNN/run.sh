#!/bin/bash
image_path='../Datasets/Image'
label_path='../Datasets/bboxes_for_SUIT_images_2021-05-10-02-29-37'
ratio=0.76
directory='../Datasets/images'
# if there is one or more args, remove the test and train datasets
if [ $# -ge 1 ]; then
  rm -r $directory
  rm -r $directory
fi
if [ ! -d $directory ]; then
  python -W ignore split_data.py --image_path $image_path --label_path $label_path --train_ratio $ratio
  python -W ignore prepare_yolo_format.py
fi

python -W ignore main.py --image_path $directory --label_path $label_path

