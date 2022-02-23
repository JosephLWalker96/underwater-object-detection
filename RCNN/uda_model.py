import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Function
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from wbf_ensemble import run_wbf, make_ensemble_predictions

'''
    The baseline model is based on fasterrcnn with a resnet50 as cnn
'''
class net(nn.Module):
    def __init__(self, num_classes, nn_type = "faster-rcnn", use_grayscale=False):
        super(net, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.nn_type = nn_type
        if self.nn_type == "faster-rcnn":
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        elif self.nn_type == 'faster-rcnn-mobilenet':
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        else:
            print("Current choice of models are: faster-rcnn")
            raise ValueError

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # for domain classifier
        self.domain_classifier = DomainClassifier(in_features) # 0 for source and 1 for target
        self.domain_loss_fuc = torch.nn.NLLLoss()

    def forward(self, images, domain_labels=None, targets=None):
        if domain_labels is None:
            domain_labels = []
        alpha = 1
        box_feature = []
        hook = None
        def extract_box_feature(module, inp, out):
            box_feature.append(inp[0].data.cpu().numpy())

        if self.model.training:
            hook = self.model.roi_heads.box_predictor.register_forward_hook(hook=extract_box_feature)

        net_out = self.model.forward(images=images, targets=targets)

        if self.model.training:
            box_feature = torch.tensor(box_feature).to(self.device)
            reverse_box_feature = ReverseLayerF.apply(box_feature, alpha)
            batch_size, num_bbox, in_features = reverse_box_feature.shape
            reverse_box_feature = reverse_box_feature.view(batch_size*num_bbox, in_features)

            domain_out = self.domain_classifier(reverse_box_feature)
            # num_box_per_image = num_bbox/batch_size = num_bbox/len(use_color_matcher)
            domain_labels = self.get_domain_labels(domain_labels, num_bbox / len(domain_labels))
            domain_loss = self.domain_loss_fuc(domain_out.to(self.device), domain_labels.to(self.device))
            print(domain_loss)
            net_out['domain_loss'] = domain_loss
            if hook is not None:
                hook.remove()
        return net_out

    def predict(self, images):
        self.eval()
        images = list(image.to(self.device) for image in images)
        predictions = make_ensemble_predictions(images, self.device, [self])
        outputs = []
        for i, image in enumerate(images):
            boxes, scores, labels = run_wbf(predictions, image_index=i)
            boxes = boxes.astype(np.int32).clip(min=0, max=512) / 512
            single_output = {'boxes': boxes, 'scores': scores, 'labels': labels}
            outputs.append(single_output)
        return outputs

    def get_domain_labels(self, domain_labels, num_bbox):
        domain_labels = torch.cat([label * torch.ones(int(num_bbox)) for label in domain_labels], 0)
        return domain_labels.view(-1).long().to(self.device)

class DomainClassifier(nn.Module):
    def __init__(self, in_features):
        super(DomainClassifier, self).__init__()
        self.domain_classifier = nn.Sequential()
        # cite from: https://github.com/fungtion/DANN/blob/master/train/main.py
        # self.domain_classifier.add_module('d_fc1', nn.Linear(in_features, 100))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        # self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        # self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        # self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

        # MLP -> SoftMax
        self.domain_classifier.add_module('d_fc1', nn.Linear(in_features, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, x):
        return self.domain_classifier(x)

# cite from: https://github.com/fungtion/DANN/blob/master/train/main.py
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None