import torch
import random
from rpnnet.anchor_generator import all_anchor_generator
from utils.boxes import filter_boundary_anchor, iou_calcu, xyxy_to_xywh, gtformat_to_regformat

import torchvision.transforms as transforms
from utils.visualization import plot_bndbox


"""
    # Algorithms:
    # generate all anchors over the grid
    # exclude anchors those are out of bound
    # calculate the overlap between anchors and ground-truth boxes
    # define positive anchors and negative anchors
    #   # each ground truth each match an anchor with max overlap
    #   # assign anchor with overlap larger than thresh to positive, less than thresh to negative
    #   # assign label
    # sampling positive and negative anchors
    #   # disable some samples, and set the related label to -1
    # assign box regression target
"""


# delta is the weight of the all picked pos
def build_anchor_targets(gt_box, im_info, feature_info, anchor_info, cfg):
    # generate all anchors over the grid
    im_height, im_width = im_info
    H, W = feature_info
    A, stride, aspect, scale = anchor_info
    anchor_boxs = all_anchor_generator(feature_info, anchor_info)
    anchor_boxs = anchor_boxs.type_as(gt_box)

    # define final return
    tar_cls = torch.zeros((H * W * A, 1)).type_as(gt_box)
    tar_boxs = torch.zeros((H * W * A, 4)).type_as(gt_box)
    # cls_weights is for a img batch(256), reg_weights is only for the pos anchor(128 is max)
    cls_weights = torch.zeros((H * W * A, 1)).type_as(gt_box)
    reg_weights = torch.zeros((H * W * A, 1)).type_as(gt_box)

    # exclude anchors those are out of bound
    anchor_boxs = torch.reshape(anchor_boxs, (H * W * A, 4))
    anchor_keep = filter_boundary_anchor(anchor_boxs, im_width, im_height)
    anchor_keep_idx = torch.where(anchor_keep)[0]
    anchor_boxs = anchor_boxs[anchor_keep_idx]

    # calculate the overlap between anchors and ground-truth boxes
    # cross_iou is K * N, K is the nums of anchor_boxs(H * W * A - cross_bound), N is the nums of gt_box
    batch_size, gt_nums, gt_points = gt_box.shape
    gt_box = torch.reshape(gt_box, (batch_size * gt_nums, gt_points))
    cross_iou = iou_calcu(anchor_boxs, gt_box)
    anchor_max, anchor_max_idx = torch.max(cross_iou, dim=1)  # k
    gt_max, gt_max_idx = torch.max(cross_iou, dim=0)  # N

    # assign a reg box for each anchor based on the max gt
    reg_box = gtformat_to_regformat(xyxy_to_xywh(gt_box[anchor_max_idx]), xyxy_to_xywh(anchor_boxs))

    # define positive anchors and negative anchors
    #   # each ground truth each match an anchor with max overlap
    #   # assign anchor with overlap larger than thresh to positive, less than thresh to negative
    #   # assign label, 0 is background, 1 is foreground
    negative_box_keep = torch.where(anchor_max < cfg['NEG_ANCHOR_TH'])[0]
    negative_box_idx = anchor_keep_idx[negative_box_keep]

    # positive_box_keep_type1 is the anchor which iou with any gt greater than pos_anchor_threshold(0.7)
    # positive_box_keep_type2 is the anchor which iou has the max overlap with one gt
    positive_box_keep_type1 = torch.where(anchor_max > cfg['POS_ANCHOR_TH'])[0]
    positive_box_keep_type2 = torch.where(torch.sum((cross_iou == gt_max), dim=1))[0]
    positive_box_keep_type = torch.zeros(anchor_keep_idx.shape[0])
    positive_box_keep_type[positive_box_keep_type1] = 1
    positive_box_keep_type[positive_box_keep_type2] = 1
    positive_box_keep = torch.where(positive_box_keep_type)[0]
    positive_box_idx = anchor_keep_idx[positive_box_keep]

    # sampling positive and negative anchors
    rois_per_img = cfg['ANCHORS_PER_IMG']
    pos_rois_num = min(cfg['ANCHORS_POS_PER_IMG'], positive_box_idx.shape[0])
    pos_picked_ids = random.sample(list(range(positive_box_idx.shape[0])), pos_rois_num)
    positive_box_idx = positive_box_idx[pos_picked_ids]
    positive_box_keep = positive_box_keep[pos_picked_ids]

    neg_roi_nums = min(rois_per_img - pos_rois_num, negative_box_idx.shape[0])
    neg_picked_ids = random.sample(list(range(negative_box_idx.shape[0])), neg_roi_nums)
    negative_box_idx = negative_box_idx[neg_picked_ids]

    # assign box cls, box regression and weights, target
    tar_cls[negative_box_idx] = 0
    tar_cls[positive_box_idx] = 1

    tar_boxs[positive_box_idx] = reg_box[positive_box_keep]

    total_box_idx = torch.cat([negative_box_idx, positive_box_idx])
    cls_weights[total_box_idx] = cfg['CLS_WEIGHT_DETLA']
    reg_weights[positive_box_idx] = cfg['REG_WEIGHT_DETLA']

    # it should be noticed that:
    # tar cls has 256 0 and 1 and others are -1
    # weights has 256 delta and others are 0
    # tar boxs has all positive nums value and others are [0,0,0,0]
    return tar_cls, tar_boxs, cls_weights, reg_weights

