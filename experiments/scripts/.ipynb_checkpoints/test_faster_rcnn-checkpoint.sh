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
    ITERS=70000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  pascal_voc_0712)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval"
    TEST_IMDB="voc_2007_test"
    ITERS=110000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train+coco_2014_valminusminival"
    TEST_IMDB="coco_2014_minival"
    ITERS=490000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  KITTI)
    TRAIN_IMDB="KITTI_train"
    TEST_IMDB="KITTI_val"
    STEPSIZE="[350000]"
    ITERS=490000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  cityscapes)
    TRAIN_IMDB="cityscapes_train+cityscapes_val"
    TEST_IMDB="bdd100k_dayval"
    STEPSIZE="[350000]"
    ITERS=110000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  foggyCity)
    TRAIN_IMDB="cityscapes_foggytrain"
    TEST_IMDB="cityscapes_foggyval"
    STEPSIZE="[350000]"
    ITERS=80000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  bdd100k)
    TRAIN_IMDB="bdd100k_nighttrain"
    TEST_IMDB="bdd100k_nightval"
    STEPSIZE="[350000]"
    ITERS=200000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA)
    TRAIN_IMDB='PAL2021_train'
    TEST_IMDB='HUA_trainval'
    ITERS=60000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL)
    TRAIN_IMDB='PAL2021_train'
    TEST_IMDB='PAL_trainval'
    ITERS=60000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  LL)
    TRAIN_IMDB='PAL2021_train'
    TEST_IMDB='LL_trainval'
    ITERS=60000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  MOO)
    TRAIN_IMDB='MOO_train'
    TEST_IMDB='PAL2021_trainval'
    ITERS=60000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  TAH)
    TRAIN_IMDB='TAH_train'
    TEST_IMDB='PAL2021_trainval'
    ITERS=60000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2021)
    TRAIN_IMDB='TAK_train'
    TEST_IMDB='PAL2021_trainval'
    ITERS=35000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  RAN)
    TRAIN_IMDB='PAL2021_train'
    TEST_IMDB='RAN_trainval'
    ITERS=60000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  TAK)
    TRAIN_IMDB='PAL2021_train'
    TEST_IMDB='TAK_trainval'
    ITERS=60000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL20212HUA)
    TRAIN_IMDB='PAL2021_train'
    TEST_IMDB='PAL2021synthHUA_val'
    ITERS=30084
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL20212LL)
    TRAIN_IMDB='PAL2021_train'
    TEST_IMDB='PAL2021synthLL_val'
    ITERS=30084
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL20212PAL)
    TRAIN_IMDB='PAL2021_train'
    TEST_IMDB='PAL2021synthPAL_val'
    ITERS=30084
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL20212TAK)
    TRAIN_IMDB='PAL2021_train'
    TEST_IMDB='PAL2021synthTAK_val'
    ITERS=30084
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL20212RAN)
    TRAIN_IMDB='PAL2021_train'
    TEST_IMDB='PAL2021synthRAN_val'
    ITERS=30084
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2PAL2021)
    TRAIN_IMDB='PAL2021_train'
    TEST_IMDB='HUAsynthPAL2021_trainval'
    ITERS=30084
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  LL2PAL2021)
    TRAIN_IMDB='PAL2021_train'
    TEST_IMDB='LLsynthPAL2021_val'
    ITERS=30084
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2PAL2021)
    TRAIN_IMDB='PAL2021_train'
    TEST_IMDB='PALsynthPAL2021_val'
    ITERS=30084
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  RAN2PAL2021)
    TRAIN_IMDB='PAL2021_train'
    TEST_IMDB='RANsynthPAL2021_val'
    ITERS=30084
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  TAK2PAL2021)
    TRAIN_IMDB='PAL2021_train'
    TEST_IMDB='TAKsynthPAL2021_val'
    ITERS=30084
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/test_${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.pth
else
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/default/${NET}_faster_rcnn_iter_${ITERS}.pth
fi
set -x

if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}.yml \
    --tag ${EXTRA_ARGS_SLUG} \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
          ${EXTRA_ARGS}
else
  CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}.yml \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
          ${EXTRA_ARGS}
fi

