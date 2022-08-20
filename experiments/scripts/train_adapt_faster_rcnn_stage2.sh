#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
ADAPT_MODE=$2
NET=$3
# PREV_ITERS=12122
# PREV_ITERS=$4
# PREV_ITERS=5510
#PREV_ITERS=11950
PREV_ITERS=11700

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${ADAPT_MODE} in
  K2C)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="KITTI_train+KITTI_val"
    TRAIN_IMDB_S="KITTI_synthCity"
    TRAIN_IMDB_T="cityscapes_train"
    TEST_IMDB="cityscapes_val"
    STEPSIZE="[]"
    ITERS=10000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  C2F)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="cityscapes_train"
    TRAIN_IMDB_S="cityscapes_synthFoggytrain"
    TRAIN_IMDB_T="cityscapes_foggytrain"
    TEST_IMDB="cityscapes_foggyval"
    STEPSIZE="[]"
    ITERS=60000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  C2BDD)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="cityscapes_train+cityscapes_val"
    TRAIN_IMDB_S="cityscapes_synthBDDdaytrain+cityscapes_synthBDDdayval"
    TRAIN_IMDB_T="bdd100k_daytrain"
    TEST_IMDB="bdd100k_dayval"
    STEPSIZE="[10000]"
    ITERS=4000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2PAL_CycleGAN)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="HUA_train+HUA_val"
    TRAIN_IMDB_S="HUA_synthPALtrain+HUA_synthPALval"
    TRAIN_IMDB_T="PAL_trainval"
    TEST_IMDB="PAL_val"
    STEPSIZE="[10000]"
    ITERS=4000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2PAL_cm)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="HUA_train+HUA_val"
    TRAIN_IMDB_S="HUAsynthPAL_train+HUAsynthPAL_val"
    TRAIN_IMDB_T="PAL_trainval"
    TEST_IMDB="PAL_val"
    STEPSIZE="[2755]"
    ITERS=5510
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2LL_CycleGAN)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="HUA_train+HUA_val"
    TRAIN_IMDB_S="HUAsynthLL_train+HUAsynthLL_val"
    TRAIN_IMDB_T="LL_trainval"
    TEST_IMDB="LL_val"
    STEPSIZE="[10000]"
    ITERS=4000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2LL_cm)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="HUA_train+HUA_val"
    TRAIN_IMDB_S="HUAsynthLL_train+HUAsynthLL_val"
    TRAIN_IMDB_T="LL_trainval"
    TEST_IMDB="LL_val"
    STEPSIZE="[2755]"
    ITERS=5510
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2TAK_CycleGAN)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="HUA_train+HUA_val"
    TRAIN_IMDB_S="HUAsynthTAK_train+HUAsynthTAK_val"
    TRAIN_IMDB_T="TAK_trainval"
    TEST_IMDB="TAK_val"
    STEPSIZE="[10000]"
    ITERS=4000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2TAK_cm)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="HUA_train+HUA_val"
    TRAIN_IMDB_S="HUAsynthTAK_train+HUAsynthTAK_val"
    TRAIN_IMDB_T="TAK_trainval"
    TEST_IMDB="TAK_val"
    STEPSIZE="[2755]"
    ITERS=5510
    PREV_ITERS=5510
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2RAN_CycleGAN)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="HUA_train+HUA_val"
    TRAIN_IMDB_S="HUAsynthRAN_train+HUAsynthRAN_val"
    TRAIN_IMDB_T="RAN_trainval"
    TEST_IMDB="RAN_val"
    STEPSIZE="[10000]"
    ITERS=4000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2RAN_cm)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="HUA_train+HUA_val"
    TRAIN_IMDB_S="HUAsynthRAN_train+HUAsynthRAN_val"
    TRAIN_IMDB_T="RAN_trainval"
    TEST_IMDB="RAN_val"
    STEPSIZE="[2755]"
    ITERS=5510
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2PAL2021_CycleGAN)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="HUA_train+HUA_val"
    TRAIN_IMDB_S="HUAsynthPAL2021_train+HUAsynthPAL2021_val"
    TRAIN_IMDB_T="PAL2021_trainval"
    TEST_IMDB="PAL2021_val"
    STEPSIZE="[10000]"
    ITERS=4000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2PAL2021_cm)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="HUA_train+HUA_val"
    TRAIN_IMDB_S="HUAsynthPAL2021_train+HUAsynthPAL2021_val"
    TRAIN_IMDB_T="PAL2021_trainval"
    TEST_IMDB="PAL2021_val"
    STEPSIZE="[2755]"
    ITERS=5510
    PREV_ITERS=7714
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
    
  PAL2HUA_cm)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="PAL_trainval"
    TRAIN_IMDB_S="PALsynthHUA_trainval"
    TRAIN_IMDB_T="HUA_trainval"
    TEST_IMDB="HUA_train+HUA_val"
    STEPSIZE="[2988]"
    ITERS=5975
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2LL_cm)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="PAL_trainval"
    TRAIN_IMDB_S="PALsynthLL_trainval"
    TRAIN_IMDB_T="LL_trainval"
    TEST_IMDB="LL_val"
    STEPSIZE="[2988]"
    ITERS=5975
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2PAL2021_cm)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="PAL_trainval"
    TRAIN_IMDB_S="PALsynthPAL2021_trainval"
    TRAIN_IMDB_T="PAL2021_trainval"
    TEST_IMDB="PAL2021_val"
    STEPSIZE="[2988]"
    ITERS=5975
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2RAN_cm)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="PAL_trainval"
    TRAIN_IMDB_S="PALsynthRAN_trainval"
    TRAIN_IMDB_T="RAN_trainval"
    TEST_IMDB="RAN_val"
    STEPSIZE="[2988]"
    ITERS=5975
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2TAK_cm)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="PAL_trainval"
    TRAIN_IMDB_S="PALsynthTAK_trainval"
    TRAIN_IMDB_T="TAK_trainval"
    TEST_IMDB="TAK_val"
    STEPSIZE="[2988]"
    ITERS=5975
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
    
  PAL20212HUA_cm)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="PAL2021_trainval"
    TRAIN_IMDB_S="PAL2021synthHUA_trainval"
    TRAIN_IMDB_T="HUA_trainval"
    TEST_IMDB="HUA_train+HUA_val"
    STEPSIZE="[2988]"
    ITERS=5975
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL20212LL_cm)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="PAL2021_trainval"
    TRAIN_IMDB_S="PAL2021synthLL_trainval"
    TRAIN_IMDB_T="LL_trainval"
    TEST_IMDB="LL_val"
    STEPSIZE="[2988]"
    ITERS=5975
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL20212PAL_cm)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="PAL2021_trainval"
    TRAIN_IMDB_S="PAL2021synthPAL_trainval"
    TRAIN_IMDB_T="PAL_trainval"
    TEST_IMDB="PAL_val"
    STEPSIZE="[2988]"
    ITERS=5975
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL20212RAN_cm)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="PAL2021_trainval"
    TRAIN_IMDB_S="PAL2021synthRAN_trainval"
    TRAIN_IMDB_T="RAN_trainval"
    TEST_IMDB="RAN_val"
    STEPSIZE="[2988]"
    ITERS=5975
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL20212TAK_cm)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="PAL2021_trainval"
    TRAIN_IMDB_S="PAL2021synthTAK_trainval"
    TRAIN_IMDB_T="TAK_trainval"
    TEST_IMDB="TAK_val"
    STEPSIZE="[2988]"
    ITERS=5975
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
      --weight output/${NET}/${PREV_S}/_adapt/${NET}_faster_rcnn_${ADAPT_MODE}_stage1_iter_${PREV_ITERS}.pth \
      --imdb ${TRAIN_IMDB_S} \
      --imdbval ${TEST_IMDB} \
      --imdb_T ${TRAIN_IMDB_T} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/${NET}_${ADAPT_MODE}.yml \
      --tag ${EXTRA_ARGS_SLUG}_adapt \
      --net ${NET} \
      --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} ADAPT_MODE ${ADAPT_MODE} \
      TRAIN.STEPSIZE ${STEPSIZE} TRAIN.SNAPSHOT_PREFIX ${SNAPSHOT_PREFIX} ${EXTRA_ARGS}
  else
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/trainval_net_adapt.py \
      --weight output/${NET}/${PREV_S}/_adapt/${NET}_faster_rcnn_${ADAPT_MODE}_stage1_iter_${PREV_ITERS}.pth \
      --imdb ${TRAIN_IMDB_S} \
      --imdbval ${TEST_IMDB} \
      --imdb_T ${TRAIN_IMDB_T} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/${NET}_${ADAPT_MODE}.yml \
      --tag ${EXTRA_ARGS_SLUG}_adapt \
      --net ${NET} \
      --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} ADAPT_MODE ${ADAPT_MODE} \
      TRAIN.STEPSIZE ${STEPSIZE} TRAIN.SNAPSHOT_PREFIX ${SNAPSHOT_PREFIX} ${EXTRA_ARGS}
  fi
fi

./experiments/scripts/test_adapt_faster_rcnn_stage2.sh $@ ${ITERS}
