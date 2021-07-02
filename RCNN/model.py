import torch.nn as nn
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

'''
    The baseline model is based on fasterrcnn with a resnet50 as cnn
'''
class net(nn.Module):
    def __init__(self, num_classes, nn_type = "faster-rcnn"):
        super(net, self).__init__()
        self.nn_type = nn_type
        if self.nn_type == "faster-rcnn":
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

            # get number of input features for the classifier
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features

            # replace the pre-trained head with a new one
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        elif self.nn_type == "retinanet":
            self.model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True, num_classes=num_classes)
        else:
            print("Current choice of models are: faster-rcnn, retinanet")
            raise ValueError

    def forward(self, images, targets=None):
        return self.model.forward(images=images, targets=targets)
