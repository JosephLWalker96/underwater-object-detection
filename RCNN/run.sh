#!/bin/bash
#rm -r ../Datasets/test_images
#rm -r ../Datasets/train_images
image_path='../Datasets/Image'
label_path='../Datasets/bboxes_for_SUIT_images_2021-05-10-02-29-37'
ratio=0.76
test_directory='../Datasets/test_images'
train_directory='../Datasets/train_images'
# if there is one or more args, remove the test and train datasets
if [ $# -ge 1 ]; then
  rm -r $test_directory
  rm -r $train_directory
fi
if [ ! -d $test_directory ]; then
  python -W ignore split_data.py --image_path $image_path --label_path $label_path --train_ratio $ratio
fi
if [ ! -d $train_directory ]; then
  python -W ignore split_data.py --image_path $image_path --label_path $label_path --train_ratio $ratio
fi
python -W ignore main.py --test_image_path $test_directory --train_image_path $train_directory --label_path $label_path

