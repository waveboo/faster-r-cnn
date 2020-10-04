import torch
import torch.nn as nn
import torch.nn.functional as F


from rpnnet.region_proposal import get_region_proposal
from rpnnet.anchor_targets import build_anchor_targets


class RPN(nn.Module):
    def __init__(self, cfg):
        super(RPN, self).__init__()
        self.n_features = cfg['MODEL']['N_FEATURES']
        self.aspect = cfg['TRAIN']['ANCHORS']['ASPECT']
        self.scale = cfg['TRAIN']['ANCHORS']['SCALE']
        self.stride = cfg['TRAIN']['ANCHORS']['STRIDE']
        self.n_anchors = len(self.aspect) * len(self.scale)

        self.cfg = cfg

        # self.out_weights_detla = cfg['TRAIN']['RPN']['OUT_WEIGHT_DETLA']

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
        batch_size, _, h, w = rpn_cls_score.shape
        rpn_cls_score_tmp = torch.reshape(rpn_cls_score, (batch_size, self.n_anchors, 2, h, w))
        rpn_cls_pred = F.softmax(rpn_cls_score_tmp, dim=2)
        rpn_cls_pred = torch.reshape(rpn_cls_pred, (batch_size, self.n_anchors*2, h, w))
        rpn_cls_pred = rpn_cls_pred.permute(0, 2, 3, 1)

        rpn_reg_score = self.rpn_reg_layer(x)
        rpn_reg_score = rpn_reg_score.permute(0, 2, 3, 1)

        key = 'TRAIN' if self.training else 'TEST'
        rpn_cfg = self.cfg[key]['RPN']

        rois = get_region_proposal(rpn_cls_pred, rpn_reg_score, im_info, (h, w),
                                   (self.n_anchors, self.stride, self.aspect, self.scale), rpn_cfg)

        if self.training:
            tar_cls, tar_boxs, cls_weights, reg_weights = build_anchor_targets(gt_box, im_info, (h, w),
                                                              (self.n_anchors, self.stride, self.aspect, self.scale),
                                                              rpn_cfg)
            tar_cls = tar_cls.long().squeeze()
            cls_weights = cls_weights.squeeze()
            reg_weights = reg_weights.squeeze()

            # cao1 = torch.nonzero(weights).shape
            # cao2 = torch.nonzero(tar_boxs[cao1]).shape
            # cao3 = torch.nonzero(tar_cls).shape

            rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1)
            rpn_cls_score = torch.reshape(rpn_cls_score, (batch_size * h * w * self.n_anchors, 2))
            rpn_reg_score = torch.reshape(rpn_reg_score, (batch_size * h * w * self.n_anchors, 4))
            cls_loss = cls_weights * F.cross_entropy(rpn_cls_score, tar_cls, reduction='none')
            cls_loss = torch.sum(cls_loss) / 256

            pos_nums = torch.nonzero(reg_weights).shape[0]
            pos_keep = torch.nonzero(reg_weights)
            rpn_reg_pos = rpn_reg_score[pos_keep]
            tar_boxs_pos = tar_boxs[pos_keep]
            # cao = F.smooth_l1_loss(rpn_reg_pos, tar_boxs_pos, reduction='none')
            # fuckfuck = F.smooth_l1_loss(rpn_reg_pos, tar_boxs_pos, reduction='mean')
            # fuck = torch.sum(torch.sum(cao, dim=1)) / pos_nums
            reg_loss = reg_weights * torch.sum(F.smooth_l1_loss(rpn_reg_score, tar_boxs, reduction='none'), dim=1)
            reg_loss = torch.sum(reg_loss) / pos_nums

            return cls_loss, reg_loss, rois
        else:
            return 0, 0, rois

