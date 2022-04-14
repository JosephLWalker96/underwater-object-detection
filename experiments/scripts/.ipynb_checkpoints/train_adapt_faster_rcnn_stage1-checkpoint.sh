#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
ADAPT_MODE=$2
NET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${ADAPT_MODE} in
  K2C)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_KITTI_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="KITTI_train+KITTI_val"
    TRAIN_IMDB_T="KITTI_synthCity"
    TEST_IMDB="cityscapes_val"
    STEPSIZE="[50000]"
    ITERS=70000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  C2F)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_city_pretrained_8class.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="cityscapes_train"
    TRAIN_IMDB_T="cityscapes_synthFoggytrain"
    TEST_IMDB="cityscapes_foggyval"
    STEPSIZE="[50000]"
    ITERS=70000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  C2BDD)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_city_pretrained_10class.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="cityscapes_train+cityscapes_val"
    TRAIN_IMDB_T="cityscapes_synthBDDdaytrain+cityscapes_synthBDDdayval"
    TEST_IMDB="bdd100k_dayval"
    STEPSIZE="[50000]"
    ITERS=70000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2PAL_CycleGAN)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_HUA_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="HUAsynthPAL_train+HUAsynthPAL_val"
    TEST_IMDB="PAL_val"
    STEPSIZE="[2755]"
    ITERS=5510
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2PAL_direct)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_HUA_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="PAL_val"
    TEST_IMDB="PAL_val"
    STEPSIZE="[2755]"
    ITERS=5510
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2PAL_cm)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_HUA_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="HUAsynthPAL_train+HUAsynthPAL_val"
    TEST_IMDB="PAL_val"
    STEPSIZE="[2755]"
    ITERS=11020
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2PAL2021_CycleGAN)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_HUA_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="HUAsynthPAL2021_train+HUAsynthPAL2021_val"
    TEST_IMDB="PAL2021_val"
    STEPSIZE="[2755]"
    ITERS=5510
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2PAL2021_direct)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_HUA_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="PAL2021_val"
    TEST_IMDB="PAL2021_val"
    STEPSIZE="[2755]"
    ITERS=5510
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2PAL2021_cm)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_HUA_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="HUAsynthPAL2021_train+HUAsynthPAL2021_val"
    TEST_IMDB="PAL2021_val"
    STEPSIZE="[2755]"
    ITERS=11020
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2TAK_CycleGAN)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_HUA_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="HUAsynthTAK_train+HUAsynthTAK_val"
    TEST_IMDB="TAK_val"
    STEPSIZE="[2755]"
    ITERS=5510
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2TAK_direct)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_HUA_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="TAK_val"
    TEST_IMDB="TAK_val"
    STEPSIZE="[2755]"
    ITERS=5510
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2TAK_cm)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_HUA_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="HUAsynthTAK_train+HUAsynthTAK_val"
    TEST_IMDB="TAK_val"
    STEPSIZE="[2755]"
    ITERS=11020
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2LL_CycleGAN)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_HUA_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="HUAsynthLL_train+HUAsynthLL_val"
    TEST_IMDB="LL_val"
    STEPSIZE="[2755]"
    ITERS=5510
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2LL_direct)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_HUA_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="LL_val"
    TEST_IMDB="LL_val"
    STEPSIZE="[2755]"
    ITERS=5510
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2LL_cm)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_HUA_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="HUAsynthLL_train+HUAsynthLL_val"
    TEST_IMDB="LL_val"
    STEPSIZE="[2755]"
    ITERS=11020
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2RAN_CycleGAN)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_HUA_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="HUAsynthRAN_train+HUAsynthRAN_val"
    TEST_IMDB="RAN_val"
    STEPSIZE="[2755]"
    ITERS=5510
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2RAN_direct)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_HUA_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="RAN_val"
    TEST_IMDB="RAN_val"
    STEPSIZE="[2755]"
    ITERS=5510
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2RAN_cm)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_HUA_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="HUAsynthRAN_train+HUAsynthRAN_val"
    TEST_IMDB="RAN_val"
    STEPSIZE="[2755]"
    ITERS=11020
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;

  PAL2HUA_CycleGAN)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_PAL_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="PAL_val"
    TRAIN_IMDB_T="PALsynthHUA_val"
    TEST_IMDB="HUA_val+HUA_train"
    STEPSIZE="[2988]"
    ITERS=5975
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2HUA_direct)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_PAL_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="PAL_val"
    TRAIN_IMDB_T="HUA_train+HUA_val"
    TEST_IMDB="HUA_val+HUA_train"
    STEPSIZE="[2988]"
    ITERS=5975
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2HUA_cm)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_PAL_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="PAL_val"
    TRAIN_IMDB_T="PALsynthHUA_val"
    TEST_IMDB="HUA_val+HUA_train"
    STEPSIZE="[2988]"
    ITERS=11950
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2LL_CycleGAN)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_PAL_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="PAL_val"
    TRAIN_IMDB_T="PALsynthLL_val"
    TEST_IMDB="LL_val"
    STEPSIZE="[2988]"
    ITERS=5975
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2LL_direct)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_PAL_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="PAL_val"
    TRAIN_IMDB_T="LL_val"
    TEST_IMDB="LL_val"
    STEPSIZE="[2988]"
    ITERS=5975
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2LL_cm)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_PAL_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="PAL_val"
    TRAIN_IMDB_T="PALsynthLL_val"
    TEST_IMDB="LL_val"
    STEPSIZE="[2988]"
    ITERS=11950
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2PAL2021_CycleGAN)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_PAL_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="PAL_val"
    TRAIN_IMDB_T="PALsynthPAL2021_val"
    TEST_IMDB="PAL2021_val"
    STEPSIZE="[2988]"
    ITERS=5975
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2PAL2021_direct)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_PAL_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="PAL_val"
    TRAIN_IMDB_T="PAL2021_val"
    TEST_IMDB="PAL2021_val"
    STEPSIZE="[2988]"
    ITERS=5975
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2PAL2021_cm)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_PAL_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="PAL_val"
    TRAIN_IMDB_T="PALsynthPAL2021_val"
    TEST_IMDB="PAL2021_val"
    STEPSIZE="[2988]"
    ITERS=11950
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2RAN_CycleGAN)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_PAL_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="PAL_val"
    TRAIN_IMDB_T="PALsynthRAN_val"
    TEST_IMDB="RAN_val"
    STEPSIZE="[2988]"
    ITERS=5975
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2RAN_direct)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_PAL_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="PAL_val"
    TRAIN_IMDB_T="RAN_val"
    TEST_IMDB="RAN_val"
    STEPSIZE="[2988]"
    ITERS=5975
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2RAN_cm)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_PAL_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="PAL_val"
    TRAIN_IMDB_T="PALsynthRAN_val"
    TEST_IMDB="RAN_val"
    STEPSIZE="[2988]"
    ITERS=11950
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2TAK_CycleGAN)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_PAL_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="PAL_val"
    TRAIN_IMDB_T="PALsynthTAK_val"
    TEST_IMDB="TAK_val"
    STEPSIZE="[2988]"
    ITERS=5975
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2TAK_direct)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_PAL_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="PAL_val"
    TRAIN_IMDB_T="TAK_val"
    TEST_IMDB="TAK_val"
    STEPSIZE="[2988]"
    ITERS=5975
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2TAK_cm)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_PAL_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="PAL_val"
    TRAIN_IMDB_T="PALsynthTAK_val"
    TEST_IMDB="TAK_val"
    STEPSIZE="[2988]"
    ITERS=11950
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/${NET}_${TRAIN_IMDB_S}2${TRAIN_IMDB_T}_${EXTRA_ARGS_SLUG}_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  NET_FINAL=output/${NET}/${TRAIN_IMDB_S}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.pth
else
  NET_FINAL=output/${NET}/${TRAIN_IMDB_S}/adapt/${NET}_faster_rcnn_iter_${ITERS}.pth
fi
set -x

if [ ! -f ${NET_FINAL}.index ]; then
  if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/trainval_net_adapt.py \
      --weight trained_weights/prerained_detector/${PRETRAINED_WEIGHT} \
      --imdb ${TRAIN_IMDB_S} \
      --imdbval ${TEST_IMDB} \
      --imdb_T ${TRAIN_IMDB_T} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/${NET}_${ADAPT_MODE}.yml \
      --tag ${EXTRA_ARGS_SLUG}_adapt \
      --net ${NET} \
      --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
      TRAIN.STEPSIZE ${STEPSIZE} TRAIN.SNAPSHOT_PREFIX ${SNAPSHOT_PREFIX} ${EXTRA_ARGS}
  else
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/trainval_net_adapt.py \
      --weight trained_weights/pretrained_detector/${PRETRAINED_WEIGHT} \
      --imdb ${TRAIN_IMDB_S} \
      --imdbval ${TEST_IMDB} \
      --imdb_T ${TRAIN_IMDB_T} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/${NET}_${ADAPT_MODE}.yml \
      --tag ${EXTRA_ARGS_SLUG}_adapt \
      --net ${NET} \
      --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
      TRAIN.STEPSIZE ${STEPSIZE} TRAIN.SNAPSHOT_PREFIX ${SNAPSHOT_PREFIX} ${EXTRA_ARGS}
  fi
fi

./experiments/scripts/test_adapt_faster_rcnn_stage1.sh $@ ${ITERS}
