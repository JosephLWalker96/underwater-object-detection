import torch
import numpy as np
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from wbf_ensemble import run_wbf, make_ensemble_predictions

'''
    The baseline model is based on fasterrcnn with a resnet50 as cnn
'''


class net(nn.Module):
    def __init__(self, num_classes, nn_type="faster-rcnn", use_grayscale=False):
        super(net, self).__init__()
        self.nn_type = nn_type
        if self.nn_type == "faster-rcnn":
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

            # get number of input features for the classifier
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features

            # replace the pre-trained head with a new one
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        elif self.nn_type == "retinanet":
            self.model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained_backbone=True,
                                                                             num_classes=num_classes)
        #             print(self.model)
        #TODO: need to debug for this one
        elif self.nn_type == 'faster-rcnn-vgg16':
            # cite: https://github.com/pytorch/vision/issues/1116#issuecomment-515373212
            vgg = torchvision.models.vgg16(pretrained=True)
            backbone = vgg.features[:-1]
            for layer in backbone[:10]:
                for p in layer.parameters():
                    p.requires_grad = False
            backbone.out_channels = 512
            anchor_generator = torchvision.models.detection.faster_rcnn.AnchorGenerator(
                sizes=((32, 64, 128, 256, 512),),
                aspect_ratios=((0.5, 1.0, 2.0),))
            roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                            output_size=7,
                                                            sampling_ratio=2)
            class BoxHead(nn.Module):
                def __init__(self, vgg):
                    super(BoxHead, self).__init__()
                    self.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

                def forward(self, x):
                    x = x.flatten(start_dim=1)
                    x = self.classifier(x)
                    return x
            box_head = BoxHead(vgg)
            self.model = torchvision.models.detection.faster_rcnn.FasterRCNN(
                backbone=backbone,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
                box_head=box_head,
                box_predictor=torchvision.models.detection.faster_rcnn.FastRCNNPredictor(4096, num_classes=3))

        elif self.nn_type == 'faster-rcnn-mobilenet':
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
            # get number of input features for the classifier
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features

            # replace the pre-trained head with a new one
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        else:
            print("Current choice of models are: faster-rcnn, retinanet")
            raise ValueError

    def forward(self, images, targets=None):
        return self.model.forward(images=images, targets=targets)

    def predict(self, images):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.eval()
        images = list(image.to(device) for image in images)
        predictions = make_ensemble_predictions(images, device, [self])
        outputs = []
        for i, image in enumerate(images):
            boxes, scores, labels = run_wbf(predictions, image_index=i)
            boxes = boxes.astype(np.int32).clip(min=0, max=512) / 512
            single_output = {'boxes': boxes, 'scores': scores, 'labels': labels}
            outputs.append(single_output)
        return outputs
