# Progressive Domain Adaptation for Object Detection (from https://github.com/kevinhkhsu/DA_detection)
```
@inproceedings{hsu2020progressivedet,
  author = {Han-Kai Hsu and Chun-Han Yao and Yi-Hsuan Tsai and Wei-Chih Hung and Hung-Yu Tseng and Maneesh Singh and Ming-Hsuan Yang},
  booktitle = {IEEE Winter Conference on Applications of Computer Vision (WACV)},
  title = {Progressive Domain Adaptation for Object Detection},
  year = {2020}
}
```


## Dependencies
This code is tested with **Pytorch 0.4.1** and **CUDA 9.0**
```
# Pytorch via pip: Download and install Pytorch 0.4.1 wheel for CUDA 9.0
#                  from https://download.pytorch.org/whl/cu90/torch_stable.html
# Pytorch via conda: 
conda install pytorch=0.4.1 cuda90 -c pytorch
# Other dependencies:
pip install -r requirements.txt
sh ./lib/make.sh
```

## Data Preparation
#### HUA
- Download the data from [here](https://drive.google.com/file/d/1qVys_q6kJsKD9fI0nEgCxMXlWsXsJIMB/view?usp=sharing).
- Extract the files under `data/`

#### LL
- Download the data from [here](https://drive.google.com/file/d/1wkJYBn7Dt-sFQLcfJxS8gvszOnkMvcw5/view?usp=sharing).
- Extract the files under `data/`

#### PAL_2020 FR
- Download the data from [here](https://drive.google.com/file/d/1XS1fks52eXOpYsH5_kGlAzlw7K6UH38M/view?usp=sharing).
- Extract the files under `data/`

#### PAL_2021 RT
- Download the data from [here](https://drive.google.com/file/d/1vp_AOVgbXS4dr_Tg1tDdAhryPvzVvijX/view?usp=sharing).
- Extract the files under `data/`

#### TAK
- Download the data from [here](https://drive.google.com/file/d/1syo7P1sa1W5WcIIyiow9BgYdFgmnPUA6/view?usp=sharing).
- Extract the files under `data/`

#### RAN
- Download the data from [here](https://drive.google.com/file/d/1syo7P1sa1W5WcIIyiow9BgYdFgmnPUA6/view?usp=sharing).
- Extract the files under `data/`

#### Preparing synthetic dataset with color-matcher:
cd data;
python ColorMatcherSynthMP.py;

## Test the adaptation model (TODO: NEED TO UPLOAD TRAINED WEIGHTS)
Download the following adapted weights to `./trained_weights/adapt_weight/`
- [TAK2HUA](https://drive.google.com/file/d/1ctnBEzk_xwPJaYAgs4IxMRzM8qafNPRi/view?usp=sharing)
- [TAK2LL](https://drive.google.com/file/d/1Pkfmf0rJCARWAtFF3ICmDj4lJaCwsPYq/view?usp=sharing)
- [TAK2PAL](https://drive.google.com/file/d/1R6Q4P9DfJCO_EFBHRf7_E4TTUYUSbcEj/view?usp=sharing)
- [TAK2PAL2021](https://drive.google.com/file/d/1xYMNg0C6hpEdldslxhJEs9Bmi91-kUMc/view?usp=sharing)
- [TAK2RAN](https://drive.google.com/file/d/1w2Opyxo5sd4LcqK9pVt4LCwbE5U50GNm/view?usp=sharing)
```
./experiments/scripts/test_adapt_faster_rcnn_stage1.sh [GPU_ID] [Adapt_mode] vgg16
# Specify the GPU_ID you want to use
# Adapt_mode selection:
#   'HUA2LL': HUA->LL
#   'HUA2PAL': HUA->PAL_2020(FR)
#   'HUA2PAL2021': HUA->PAL_2021(RT)
#   'HUA2TAK': HUA->TAK
#   'HUA2RAN': HUA->RAN
#   ...
# Example:
./experiments/scripts/test_adapt_faster_rcnn_stage2.sh 0 HUA2LL vgg16
```

## Train your own model
#### Stage one
```
./experiments/scripts/train_adapt_faster_rcnn_stage1.sh [GPU_ID] [Adapt_mode] vgg16
# Specify the GPU_ID you want to use
# Adapt_mode selection:
#   'HUA2LL': HUA->LL
#   'HUA2PAL': HUA->PAL_2020(FR)
#   'HUA2PAL2021': HUA->PAL_2021(RT)
#   'HUA2TAK': HUA->TAK
#   'HUA2RAN': HUA->RAN
#   ...
# Example:
./experiments/scripts/train_adapt_faster_rcnn_stage1.sh 0 HUA2LL vgg16
```
Download the following pretrained detector weights to `./trained_weights/pretrained_detector/` (the provided pretrained_detector are using TAK as source environment)
- [pretrained on images from TAK environments](https://drive.google.com/file/d/1Rv6yP0Wcg_kSCpfl4zSxURCdWqBFU9Hw/view?usp=sharing）

#### Stage two
```
./experiments/scripts/train_adapt_faster_rcnn_stage2.sh 0 HUA2LL vgg16
```

## Detection results
![](figure/det_results.png)

## Adaptation results
![](figure/adapt_results_k2c.png)
![](figure/adapt_results_c2f.png)
![](figure/adapt_results_c2bdd.png)

## Acknowledgement
We would like to express our thanks to the awesome implementations from [DA_detection](https://github.com/kevinhkhsu/DA_detection) and [pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/README.md).
