#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
ADAPT_MODE=$2
NET=$3
TEST_ITER=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${ADAPT_MODE} in
  K2C)
    TRAIN_IMDB_S="KITTI_train+KITTI_val"
    TRAIN_IMDB_T="KITTI_synthCity"
    TEST_IMDB="cityscapes_val"
    STEPSIZE="[50000]"
    ITERS=${TEST_ITER}
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  C2F)
    TRAIN_IMDB_S="cityscapes_train"
    TRAIN_IMDB_T="cityscapes_synthFoggytrain"
    TEST_IMDB="cityscapes_foggyval"
    STEPSIZE="[50000]"
    ITERS=${TEST_ITER}
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  C2BDD)
    TRAIN_IMDB_S="cityscapes_train+cityscapes_val"
    TRAIN_IMDB_T="cityscapes_synthBDDdaytrain+cityscapes_synthBDDdayval"
    TEST_IMDB="bdd100k_dayval"
    STEPSIZE="[50000]"
    ITERS=${TEST_ITER}
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2PAL_CycleGAN)
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="HUA_synthPALtrain+HUA_synthPALval"
    TEST_IMDB="PAL_val"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2PAL_direct)
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="PAL_val"
    TEST_IMDB="PAL_val"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2PAL_cm)
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="HUAsynth_PALtrain+HUAsynthPAL_val"
    TEST_IMDB="PAL_val"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2LL_CycleGAN)
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="HUAsynthLL_train+HUAsynthLL_val"
    TEST_IMDB="LL_val"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2LL_direct)
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="LL_val"
    TEST_IMDB="LL_val"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2LL_cm)
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="HUAsynthLL_train+HUAsynthLL_val"
    TEST_IMDB="LL_val"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2PAL2021_CycleGAN)
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="HUAsynthPAL2021_train+HUAsynthPAL2021_val"
    TEST_IMDB="PAL2021_val"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2PAL2021_direct)
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="PAL2021_val"
    TEST_IMDB="PAL2021_val"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2PAL2021_cm)
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="HUAsynthPAL_2021train+HUAsynthPAL2021_val"
    TEST_IMDB="PAL2021_val"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2TAK_CycleGAN)
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="HUAsynthTAK_train+HUAsynthTAK_val"
    TEST_IMDB="TAK_val"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2TAK_direct)
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="TAK_val"
    TEST_IMDB="TAK_val"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2TAK_cm)
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="HUAsynthTAK_train+HUAsynthTAK_val"
    TEST_IMDB="TAK_val"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2RAN_CycleGAN)
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="HUAsynthRAN_train+HUAsynthRAN_val"
    TEST_IMDB="RAN_val"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2RAN_direct)
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="RAN_val"
    TEST_IMDB="RAN_val"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2RAN_cm)
    TRAIN_IMDB_S="HUA_train+HUA_val"
    TRAIN_IMDB_T="HUAsynthRAN_train+HUAsynthRAN_val"
    TEST_IMDB="RAN_val"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
    
  PAL2HUA_direct)
    TRAIN_IMDB_S="PAL_val"
    TRAIN_IMDB_T="HUA_train+HUA_val"
    TEST_IMDB="HUA_trainval"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2HUA_cm)
    TRAIN_IMDB_S="PAL_val"
    TRAIN_IMDB_T="PALsynthHUA_val"
    TEST_IMDB="HUA_trainval"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2LL_direct)
    TRAIN_IMDB_S="PAL_val"
    TRAIN_IMDB_T="LL_val"
    TEST_IMDB="LL_val"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2LL_cm)
    TRAIN_IMDB_S="PAL_val"
    TRAIN_IMDB_T="PALsynthLL_val"
    TEST_IMDB="LL_val"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2PAL2021_direct)
    TRAIN_IMDB_S="PAL_val"
    TRAIN_IMDB_T="PAL2021_val"
    TEST_IMDB="PAL2021_val"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2PAL2021_cm)
    TRAIN_IMDB_S="PAL_val"
    TRAIN_IMDB_T="PALsynthPAL2021_val"
    TEST_IMDB="PAL2021_val"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2TAK_direct)
    TRAIN_IMDB_S="PAL_val"
    TRAIN_IMDB_T="TAK_val"
    TEST_IMDB="TAK_val"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2TAK_cm)
    TRAIN_IMDB_S="PAL_val"
    TRAIN_IMDB_T="PALsynthTAK_val"
    TEST_IMDB="TAK_val"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2RAN_direct)
    TRAIN_IMDB_S="PAL_val"
    TRAIN_IMDB_T="RAN_val"
    TEST_IMDB="RAN_val"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2RAN_cm)
    TRAIN_IMDB_S="PAL_val"
    TRAIN_IMDB_T="PALsynthRAN_val"
    TEST_IMDB="RAN_val"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/test_${NET}_${TRAIN_IMDB_S}_adapt_${TEST_IMDB}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  NET_FINAL=output/${NET}/${TRAIN_IMDB_S}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.pth
else
  NET_FINAL=output/${NET}/${TRAIN_IMDB_S}/_adapt/${NET}_faster_rcnn_${ADAPT_MODE}_stage1_iter_${ITERS}.pth
fi

if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}_${ADAPT_MODE}.yml \
    --tag ${EXTRA_ARGS_SLUG}_adapt \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
          ${EXTRA_ARGS}
else
  CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}_${ADAPT_MODE}.yml \
    --tag ${EXTRA_ARGS_SLUG}_adapt \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
          ${EXTRA_ARGS}
fi

