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
#### KITTI
- Download the data from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d).
- Extract the files under `data/KITTI/`

#### Cityscapes
- Download the data from [here](https://www.cityscapes-dataset.com/).
- Extract the files under `data/CityScapes/`

#### Foggy Cityscapes
- Follow the instructions [here](https://www.cityscapes-dataset.com/) to request for the dataset download.
- Locate the data under `data/CityScapes/leftImg8bit/` as `foggytrain` and `foggyval`.

#### BDD100k
- Download the data from [here](https://bdd-data.berkeley.edu/).
- Extract the files under `data/bdd100k/`

#### Train your own CycleGAN:
Please follow the training instructions on [PyTorch-CycleGAN](https://github.com/aitorzip/PyTorch-CycleGAN).

## Test the adaptation model
Download the following adapted weights to `./trained_weights/adapt_weight/`
- [KITTI->Cityscapes](http://vllab1.ucmerced.edu/~hhsu22/da_det/adapt_weight/vgg16_faster_rcnn_K2C_stage2.pth)
- [Cityscapes->FoggyCityscapes](http://vllab1.ucmerced.edu/~hhsu22/da_det/adapt_weight/vgg16_faster_rcnn_C2F_stage2.pth)
- [Cityscpaes->BDD100k](http://vllab1.ucmerced.edu/~hhsu22/da_det/adapt_weight/vgg16_faster_rcnn_C2BDD_stage2.pth)
```
./experiments/scripts/test_adapt_faster_rcnn_stage1.sh [GPU_ID] [Adapt_mode] vgg16
# Specify the GPU_ID you want to use
# Adapt_mode selection:
#   'K2C': KITTI->Cityscapes
#   'C2F': Cityscapes->Foggy Cityscapes
#   'C2BDD': Cityscapes->BDD100k_day
# Example:
./experiments/scripts/test_adapt_faster_rcnn_stage2.sh 0 K2C vgg16
```

## Train your own model
#### Stage one
```
./experiments/scripts/train_adapt_faster_rcnn_stage1.sh [GPU_ID] [Adapt_mode] vgg16
# Specify the GPU_ID you want to use
# Adapt_mode selection:
#   'K2C': KITTI->Cityscapes
#   'C2F': Cityscapes->Foggy Cityscapes
#   'C2BDD': Cityscapes->BDD100k_day
# Example:
./experiments/scripts/train_adapt_faster_rcnn_stage1.sh 0 K2C vgg16
```
Download the following pretrained detector weights to `./trained_weights/pretrained_detector/`
- [KITTI for K2C](http://vllab1.ucmerced.edu/~hhsu22/da_det/pretrained_detector/vgg16_faster_rcnn_KITTI_pretrained.pth)
- [Cityscapes for C2f](http://vllab1.ucmerced.edu/~hhsu22/da_det/pretrained_detector/vgg16_faster_rcnn_city_pretrained_8class.pth)
- [Cityscapes for C2BDD](http://vllab1.ucmerced.edu/~hhsu22/da_det/pretrained_detector/vgg16_faster_rcnn_city_pretrained_10class.pth)

#### Stage two
```
./experiments/scripts/train_adapt_faster_rcnn_stage2.sh 0 K2C vgg16
```
Discriminator score files: 
- netD_synthC_score.json
- netD_CsynthFoggyC_score.json
- netD_CsynthBDDday_score.json

Extract the pretrained [CycleGAN discriminator scores](http://vllab1.ucmerced.edu/~hhsu22/da_det/D_score.tar.gz) to `./trained_weights/` </br>
or </br>
Save a dictionary of CycleGAN discriminator scores with image name as key and score as value </br>
Ex: {'jena_000074_000019_leftImg8bit.png': 0.64}

## Detection results
![](figure/det_results.png)

## Adaptation results
![](figure/adapt_results_k2c.png)
![](figure/adapt_results_c2f.png)
![](figure/adapt_results_c2bdd.png)

## Acknowledgement
Thanks to the awesome implementations from [pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/README.md) and [PyTorch-CycleGAN](https://github.com/aitorzip/PyTorch-CycleGAN).
