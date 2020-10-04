import torch
import torch.nn as nn
from backbone.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from rpnnet.rpn import RPN
from data import FRDataLoader
import backbone as models
# from configs import backbone

from utils.visualization import plot_bndbox
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


class FASTER_RCNN(nn.Module):
    def __init__(self, cfg):
        super(FASTER_RCNN, self).__init__()
        self.cfg = cfg
        backbone = cfg['MODEL']['BACKBONE']
        assert backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        base_model = models.__dict__[backbone](pretrained=cfg['MODEL']['PRE_TRAIN'])

        self.backbone = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool,
                                      base_model.layer1, base_model.layer2, base_model.layer3)

        self.rpn = RPN(cfg)

        # Fix blocks
        for p in self.backbone[0].parameters(): p.requires_grad = False
        for p in self.backbone[1].parameters(): p.requires_grad = False
        for p in self.backbone[4].parameters(): p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.backbone.apply(set_bn_fix)
        # self.classifier.apply(set_bn_fix)

        # initial weights
        normal_init(self.rpn.rpn_feat_layer, 0, 0.01, False)
        normal_init(self.rpn.rpn_cls_layer, 0, 0.01, False)
        normal_init(self.rpn.rpn_reg_layer, 0, 0.01, False)

    def forward(self, input, gt_box, gt_cls):
        batch_size, channels, h, w = input.shape
        assert batch_size == 1, 'Only support batch_size = 1'
        x = self.backbone(input)
        cls_loss, reg_loss, rois = self.rpn(x, gt_box, (h, w))
        return cls_loss, reg_loss, rois

