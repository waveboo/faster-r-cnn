import torch
import random
from utils.boxes import iou_calcu, xyxy_to_xywh, gtformat_to_regformat

import torchvision.transforms as transforms
from utils.visualization import plot_bndbox

"""
    # Algorithm
    # merger gt boxes to all_rois
    # calculate the overlap between all rois and gt_boxes
    # sampling rois
    #     # define positive and negative rois
    #     # sample positive and negative rois if there are too many
    # assign label target
    # compute boxes target
    # assign weights
"""


def get_region_proposal(rois, gt_box, gt_cls, cls_nums, cfg):
    # define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # merger gt boxes to all_rois
    batch_size, gt_nums, gt_points = gt_box.shape
    gt_box = torch.reshape(gt_box, (batch_size * gt_nums, gt_points))
    gt_cls = gt_cls.squeeze(0)
    gt_box_tmp = torch.zeros((batch_size * gt_nums, 5)).to(device)
    gt_box_tmp[:, 1:] = gt_box
    all_boxes = torch.cat([gt_box_tmp, rois], dim=0)

    # calculate the overlap between all rois and gt_boxes
    cross_iou = iou_calcu(all_boxes[:, 1:], gt_box)
    roi_max, roi_max_idx = torch.max(cross_iou, dim=1)

    # sampling rois
    #     # define positive and negative rois
    #     # sample positive and negative rois if there are too many
    pos_roi_ids = torch.where(roi_max >= cfg['POS_ROI_TH'])[0]

    # show_pos = all_boxes[pos_roi_ids][:, 1:]
    # img = transforms.ToPILImage()(img[0].cpu())
    # plot_bndbox(img, show_pos.cpu())

    neg_roi_ids = torch.where((cfg['POS_ROI_TH'] > roi_max) & (roi_max >= cfg['NEG_ROI_TH']))[0]

    # img = transforms.ToPILImage()(img[0].cpu())
    # show_pos = all_boxes[pos_roi_ids]
    # plot_bndbox(img, show_pos[:, 1:].cpu())

    rois_per_img = cfg['ROIS_PER_IMG']
    pos_rois_num = min(int(rois_per_img * cfg['POS_FRACTION']), pos_roi_ids.shape[0])
    pos_picked_ids = random.sample(list(range(pos_roi_ids.shape[0])), pos_rois_num)
    pos_roi_keep = pos_roi_ids[pos_picked_ids]

    neg_roi_nums = rois_per_img - pos_rois_num
    if neg_roi_nums >= neg_roi_ids.shape[0]:
        neg_roi_keep = neg_roi_ids
    else:
        neg_picked_ids = random.sample(list(range(neg_roi_ids.shape[0])), neg_roi_nums)
        neg_roi_keep = neg_roi_ids[neg_picked_ids]

    all_keep = torch.cat([pos_roi_keep, neg_roi_keep], dim=0)
    rcnn_rois = all_boxes[all_keep]

    pos_labels = roi_max_idx[pos_roi_keep]
    # assign label target
    # rcnn_labels = torch.zeros((all_keep.shape[0], cls_nums)).to(device)
    # rcnn_labels[:pos_rois_num] = rcnn_labels[:pos_rois_num].scatter_(1, gt_cls[pos_labels].unsqueeze(1).long(), 1)
    rcnn_labels = torch.zeros((all_keep.shape[0], 1)).to(device)
    rcnn_labels[:pos_rois_num] = gt_cls[pos_labels].unsqueeze(1)

    # compute boxes target
    rcnn_boxes = torch.zeros((all_keep.shape[0], cls_nums * 4)).to(device)
    rcnn_boxes_weight = torch.zeros((all_keep.shape[0], cls_nums * 4)).to(device)
    reg_box = gtformat_to_regformat(xyxy_to_xywh(gt_box[pos_labels]), xyxy_to_xywh(rcnn_rois[:pos_rois_num, 1:]))

    bbox_normalize_means = torch.FloatTensor(cfg['BBOX_NORMALIZE_MEANS']).type_as(reg_box)
    bbox_normalize_stds = torch.FloatTensor(cfg['BBOX_NORMALIZE_STDS']).type_as(reg_box)

    # Normalize regression target
    if cfg['BBOX_NORMALIZE_TARGETS']:
        reg_box = (reg_box - bbox_normalize_means.view(1, 4)) \
                            / bbox_normalize_stds.view(1, 4)  # use broad-casting (1, 4) -> (N, 4)

    # make boxes target data class specific
    for i in range(pos_rois_num):
        cls_idx = gt_cls[pos_labels][i].long()
        start = cls_idx * 4
        end = start + 4
        rcnn_boxes[i, start:end] = reg_box[i]
        rcnn_boxes_weight[i, start:end] = torch.Tensor([1, 1, 1, 1]).to(device)

    # assign weights
    rcnn_weights = torch.zeros((all_keep.shape[0], 1)).to(device)
    rcnn_weights[:pos_rois_num] = cfg['REG_WEIGHT_DETLA']

    return rcnn_labels, rcnn_boxes, rcnn_weights, rcnn_boxes_weight, rcnn_rois
