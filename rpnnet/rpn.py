import torch
import torch.nn as nn
import torch.nn.functional as F


from rpnnet.region_proposal import get_region_proposal
from rpnnet.anchor_targets import build_anchor_targets


import torchvision.transforms as transforms
from utils.visualization import plot_bndbox


class RPN(nn.Module):
    def __init__(self, cfg):
        super(RPN, self).__init__()
        self.n_features = cfg['MODEL']['N_FEATURES']
        self.aspect = cfg['ANCHORS']['ASPECT']
        self.scale = cfg['ANCHORS']['SCALE']
        self.stride = cfg['ANCHORS']['STRIDE']
        self.n_anchors = len(self.aspect) * len(self.scale)

        self.cfg = cfg

        self.rpn_feat_layer = nn.Conv2d(self.n_features, 512, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.rpn_cls_layer = nn.Conv2d(512, self.n_anchors*2, kernel_size=1, stride=1, padding=0)
        self.rpn_reg_layer = nn.Conv2d(512, self.n_anchors*4, kernel_size=1, stride=1, padding=0)

    # input is the features from backbone
    # gt_box is the truth boxs
    # im_info = (img_width, img_height)
    def forward(self, input, gt_box, im_info):
        x = self.rpn_feat_layer(input)
        x = self.relu(x)
        rpn_cls_score = self.rpn_cls_layer(x)
        rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1)

        rpn_reg_score = self.rpn_reg_layer(x)
        rpn_reg_score = rpn_reg_score.permute(0, 2, 3, 1)

        batch_size, h, w, _ = rpn_cls_score.shape
        rpn_cls_score_tmp = rpn_cls_score.view(batch_size, h, w, self.n_anchors, 2)
        rpn_cls_pred = F.softmax(rpn_cls_score_tmp, dim=-1)
        rpn_cls_pred = rpn_cls_pred.view(batch_size, h, w, self.n_anchors*2)

        key = 'TRAIN' if self.training else 'TEST'
        rpn_cfg = self.cfg[key]['RPN']

        rois = get_region_proposal(rpn_cls_pred, rpn_reg_score, im_info, (h, w),
                                   (self.n_anchors, self.stride, self.aspect, self.scale), rpn_cfg)

        if self.training:
            tar_cls, tar_boxs, cls_weights, reg_weights = build_anchor_targets(gt_box, im_info, (h, w),
                                                                               (self.n_anchors, self.stride,
                                                                                self.aspect, self.scale),
                                                                               rpn_cfg)

            tar_cls = tar_cls.long().squeeze(-1)
            cls_weights = cls_weights.squeeze(-1)
            reg_weights = reg_weights.squeeze(-1)

            rpn_cls_score = torch.reshape(rpn_cls_score, (batch_size * h * w * self.n_anchors, 2))
            rpn_reg_score = torch.reshape(rpn_reg_score, (batch_size * h * w * self.n_anchors, 4))

            all_nums = torch.where(cls_weights)[0].shape[0]
            cls_loss = cls_weights * F.cross_entropy(rpn_cls_score, tar_cls, reduction='none')
            cls_loss = torch.sum(cls_loss) / all_nums

            pos_nums = torch.where(reg_weights)[0].shape[0]
            reg_loss = reg_weights * torch.sum(F.smooth_l1_loss(rpn_reg_score, tar_boxs, reduction='none'), dim=1)
            reg_loss = torch.sum(reg_loss) / pos_nums

            return cls_loss, reg_loss, rois
        else:
            return 0, 0, rois

