#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE="[50000]"
    ITERS=70000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  pascal_voc_0712)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE="[80000]"
    ITERS=110000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train+coco_2014_valminusminival"
    TEST_IMDB="coco_2014_minival"
    STEPSIZE="[350000]"
    ITERS=490000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  KITTI)
    TRAIN_IMDB="KITTI_train"
    TEST_IMDB="KITTI_val"
    STEPSIZE="[50000]"
    ITERS=70000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  cityscapes)
    TRAIN_IMDB="cityscapes_train+cityscapes_val"
    TEST_IMDB="cityscapes_val"
    STEPSIZE="[80000]"
    ITERS=110000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  foggyCity)
    TRAIN_IMDB="cityscapes_foggytrain"
    TEST_IMDB="cityscapes_foggyval"
    STEPSIZE="[80000]"
    ITERS=110000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  bdd100k)
    TRAIN_IMDB='bdd100k_daytrain'
    TEST_IMDB='bdd100k_dayval'
    STEPSIZE="[80000]"
    ITERS=110000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  sim10k)
    TRAIN_IMDB='sim10k_trainval'
    TEST_IMDB='sim10k_trainval'
    STEPSIZE="[50000]"
    ITERS=70000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA)
    TRAIN_IMDB='HUA_train'
    TEST_IMDB='HUA_val'
    STEPSIZE="[5000]"
    ITERS=60000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL)
    TRAIN_IMDB='PAL_train'
    TEST_IMDB='PAL_val'
    STEPSIZE="[5000]"
    ITERS=60000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2021)
    TRAIN_IMDB='PAL2021_train'
    TEST_IMDB='PAL2021_val'
    STEPSIZE="[5000]"
    ITERS=60000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  TAK)
    TRAIN_IMDB='TAK_train'
    TEST_IMDB='TAK_val'
    STEPSIZE="[5000]"
    ITERS=35000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  TAH)
    TRAIN_IMDB='TAH_train'
    TEST_IMDB='TAH_val'
    STEPSIZE="[5000]"
    ITERS=60000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  MOO)
    TRAIN_IMDB='MOO_train'
    TEST_IMDB='MOO_val'
    STEPSIZE="[5000]"
    ITERS=60000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUAsynthPAL)
    TRAIN_IMDB='HUAsynthPAL_train+HUAsynthPAL_val'
    TEST_IMDB='PAL_val'
    STEPSIZE="[50000]"
    ITERS=70000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.pth
else
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/default/${NET}_faster_rcnn_iter_${ITERS}.pth
fi
set -x

if [ ! -f ${NET_FINAL}.index ]; then
  if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/trainval_net.py \
      --weight data/imagenet_weights/${NET}.pth \
      --imdb ${TRAIN_IMDB} \
      --imdbval ${TEST_IMDB} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/${NET}.yml \
      --tag ${EXTRA_ARGS_SLUG} \
      --net ${NET} \
      --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
      TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
  else
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/trainval_net.py \
      --weight data/imagenet_weights/${NET}.pth \
      --imdb ${TRAIN_IMDB} \
      --imdbval ${TEST_IMDB} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/${NET}.yml \
      --net ${NET} \
      --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
      TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
  fi
fi

./experiments/scripts/test_faster_rcnn.sh $@
