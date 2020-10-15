import torch
import torch.nn as nn
import torch.nn.functional as F
from rpnnet.rpn import RPN
from rpnnet.target_propoasl import get_region_proposal
import backbone as models
from torchvision.ops import RoIPool

from utils.visualization import plot_bndbox
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utils.boxes import xyxy_to_xywh, xywh_to_xyxy, regformat_to_gtformat, clip_img_boundary
from utils.nms import nms
import numpy as np


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

        # self.classifier = nn.Sequential(base_model.layer4, base_model.avgpool)
        self.classifier = nn.Sequential(base_model.layer4)

        self.rpn = RPN(cfg)

        self.roi_pooling = RoIPool((cfg['MODEL']['POOLING_SIZE'], cfg['MODEL']['POOLING_SIZE']),
                                   1/cfg['ANCHORS']['STRIDE'])

        self.class_nums = len(cfg['DATASET']['CLASSES'])
        self.rcnn_cls_layer = nn.Linear(2*cfg['MODEL']['N_FEATURES'], self.class_nums)
        self.rcnn_reg_layer = nn.Linear(2*cfg['MODEL']['N_FEATURES'], self.class_nums*4)

        # Fix blocks
        for p in self.backbone[0].parameters(): p.requires_grad = False
        for p in self.backbone[1].parameters(): p.requires_grad = False
        for p in self.backbone[4].parameters(): p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.backbone.apply(set_bn_fix)
        self.classifier.apply(set_bn_fix)

        # initial weights
        normal_init(self.rpn.rpn_feat_layer, 0, 0.01, False)
        normal_init(self.rpn.rpn_cls_layer, 0, 0.01, False)
        normal_init(self.rpn.rpn_reg_layer, 0, 0.01, False)
        normal_init(self.rcnn_cls_layer, 0, 0.01, False)
        normal_init(self.rcnn_reg_layer, 0, 0.001, False)

    def forward(self, input, gt_box, gt_cls):
        batch_size, channels, h, w = input.shape
        assert batch_size == 1, 'Only support batch_size = 1'
        x = self.backbone(input)
        rpn_cls_loss, rpn_reg_loss, rpn_rois = self.rpn(x, gt_box, (h, w))

        if self.training:
            # key = 'TRAIN' if self.training else 'TEST'
            rcnn_cfg = self.cfg['TRAIN']['FAST_RCNN']

            rcnn_labels, rcnn_boxes, rcnn_weights, rcnn_boxes_weight, rcnn_rois = get_region_proposal(rpn_rois, gt_box,
                                                                                                      gt_cls,
                                                                                                      self.class_nums,
                                                                                                      rcnn_cfg)
        else:
            rcnn_rois = rpn_rois

        x = self.roi_pooling(x, rcnn_rois)
        x = self.classifier(x).mean(3).mean(2)

        rcnn_cls_score = self.rcnn_cls_layer(x)
        rcnn_reg_score = self.rcnn_reg_layer(x)
        rcnn_cls_prob = F.softmax(rcnn_cls_score, dim=1)

        if self.training:
            rcnn_cls_loss = F.cross_entropy(rcnn_cls_score, rcnn_labels.squeeze(-1).long())

            pos_nums = torch.where(rcnn_weights)[0].shape[0]
            rcnn_reg_loss = torch.sum(rcnn_boxes_weight *
                                      F.smooth_l1_loss(rcnn_reg_score, rcnn_boxes, reduction='none'), dim=1)
            rcnn_reg_loss = torch.sum(rcnn_reg_loss) / pos_nums

            # For train RPN:
            # return rpn_cls_loss, rpn_reg_loss, rpn_rois
            return rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss, rcnn_cls_prob, rcnn_rois
        else:
            return rcnn_cls_score, rcnn_reg_score, rcnn_cls_prob, rcnn_rois

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode:
            self.backbone.eval()
            self.backbone[5].train()
            self.backbone[6].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.backbone.apply(set_bn_eval)
            self.classifier.apply(set_bn_eval)
