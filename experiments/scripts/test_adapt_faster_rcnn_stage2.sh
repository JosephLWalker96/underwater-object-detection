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
    TRAIN_IMDB_S="KITTI_synthCity"
    TRAIN_IMDB_T="cityscapes_train"
    TEST_IMDB="cityscapes_val"
    STEPSIZE="[50000]"
    ITERS=${TEST_ITER}
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  C2F)
    TRAIN_IMDB_S="cityscapes_synthFoggytrain"
    TRAIN_IMDB_T="cityscapes_foggytrain"
    TEST_IMDB="cityscapes_foggyval"
    STEPSIZE="[50000]"
    ITERS=${TEST_ITER}
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  C2BDD)
    TRAIN_IMDB_S="cityscapes_synthBDDdaytrain+cityscapes_synthBDDdayval"
    TRAIN_IMDB_T="bdd100k_train"
    TEST_IMDB="bdd100k_dayval"
    STEPSIZE="[50000]"
    ITERS=${TEST_ITER}
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2PAL_CycleGAN)
    TRAIN_IMDB_S="HUA_synthPALtrain+HUA_synthPALval"
    TRAIN_IMDB_T="PAL_trainval"
    TEST_IMDB="PAL_trainval"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2PAL_cm)
    TRAIN_IMDB_S="HUAsynthPAL_train+HUAsynthPAL_val"
    TRAIN_IMDB_T="PAL_trainval"
    TEST_IMDB="PAL_trainval"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2RAN_CycleGAN)
    TRAIN_IMDB_S="HUA_synthRANtrain+HUA_synthRANval"
    TRAIN_IMDB_T="PAL_trainval"
    TEST_IMDB="RAN_trainval"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2RAN_cm)
    TRAIN_IMDB_S="HUAsynthRAN_train+HUAsynthRAN_val"
    TRAIN_IMDB_T="PAL_trainval"
    TEST_IMDB="RAN_trainval"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2PAL2021_CycleGAN)
    TRAIN_IMDB_S="HUAsynthPAL2021_train+HUAsynthPAL2021_val"
    TRAIN_IMDB_T="PAL2021_trainval"
    TEST_IMDB="PAL2021_trainval"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2PAL2021_cm)
    TRAIN_IMDB_S="HUAsynthPAL2021_train+HUAsynthPAL2021_val"
    TRAIN_IMDB_T="PAL2021_trainval"
    TEST_IMDB="PAL2021_trainval"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2LL_CycleGAN)
    TRAIN_IMDB_S="HUAsynthLL_train+HUAsynthLL_val"
    TRAIN_IMDB_T="LL_trainval"
    TEST_IMDB="LL_trainval"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2LL_cm)
    TRAIN_IMDB_S="HUAsynthLL_train+HUAsynthLL_val"
    TRAIN_IMDB_T="LL_trainval"
    TEST_IMDB="LL_trainval"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2TAK_CycleGAN)
    TRAIN_IMDB_S="HUAsynthTAK_train+HUAsynthTAK_val"
    TRAIN_IMDB_T="TAK_trainval"
    TEST_IMDB="TAK_trainval"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2TAK_cm)
    TRAIN_IMDB_S="HUAsynthTAK_train+HUAsynthTAK_val"
    TRAIN_IMDB_T="TAK_trainval"
    TEST_IMDB="TAK_trainval"
    ITERS=${TEST_ITER}
#     ITERS=2204
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2LL_CycleGAN)
    TRAIN_IMDB_S="HUAsynthLL_train+HUAsynthLL_val"
    TRAIN_IMDB_T="LL_trainval"
    TEST_IMDB="LL_trainval"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  HUA2LL_cm)
    TRAIN_IMDB_S="HUAsynthLL_train+HUAsynthLL_val"
    TRAIN_IMDB_T="LL_trainval"
    TEST_IMDB="LL_trainval"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;

  PAL2HUA_cm)
    TRAIN_IMDB_S="PALsynthHUA_trainval"
    TRAIN_IMDB_T="HUA_trainval"
    TEST_IMDB="HUA_trainval"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2LL_cm)
    TRAIN_IMDB_S="PALsynthLL_trainval"
    TRAIN_IMDB_T="LL_trainval"
    TEST_IMDB="LL_trainval"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2PAL2021_cm)
    TRAIN_IMDB_S="PALsynthPAL2021_trainval"
    TRAIN_IMDB_T="PAL2021_trainval"
    TEST_IMDB="PAL2021_trainval"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2RAN_cm)
    TRAIN_IMDB_S="PALsynthRAN_trainval"
    TRAIN_IMDB_T="RAN_trainval"
    TEST_IMDB="RAN_trainval"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL2TAK_cm)
    TRAIN_IMDB_S="PALsynthTAK_trainval"
    TRAIN_IMDB_T="TAK_trainval"
    TEST_IMDB="TAK_trainval"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
    
  PAL20212HUA_cm)
    TRAIN_IMDB_S="PAL2021synthHUA_trainval"
    TRAIN_IMDB_T="HUA_trainval"
    TEST_IMDB="HUA_trainval"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL20212LL_cm)
    TRAIN_IMDB_S="PAL2021synthLL_trainval"
    TRAIN_IMDB_T="LL_trainval"
    TEST_IMDB="LL_trainval"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL20212PAL_cm)
    TRAIN_IMDB_S="PAL2021synthPAL_trainval"
    TRAIN_IMDB_T="PAL_trainval"
    TEST_IMDB="PAL_trainval"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL20212RAN_cm)
    TRAIN_IMDB_S="PAL2021synthRAN_trainval"
    TRAIN_IMDB_T="RAN_trainval"
    TEST_IMDB="RAN_trainval"
    ITERS=${TEST_ITER}
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  PAL20212TAK_cm)
    TRAIN_IMDB_S="PAL2021synthTAK_trainval"
    TRAIN_IMDB_T="TAK_trainval"
    TEST_IMDB="TAK_trainval"
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
  NET_FINAL=output/${NET}/${TRAIN_IMDB_S}/_adapt/${NET}_faster_rcnn_${ADAPT_MODE}_stage2_iter_${ITERS}.pth
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

