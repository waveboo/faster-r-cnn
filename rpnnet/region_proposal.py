import torch
from torchvision.ops import nms
from rpnnet.anchor_generator import all_anchor_generator
from utils.nms import nms
from utils.boxes import xyxy_to_xywh, xywh_to_xyxy, regformat_to_gtformat, clip_img_boundary, filter_minsize_proposal

# from utils.visualization import plot_bndbox
# from utils.nms import nms
# import torchvision.transforms as transforms

"""
    Generate proposals by using rpn output prediction and pre-defined anchors.

    # Algorithm:
    # generate all anchors over the grid
    # apply delta to anchors to generate boxes (proposals)
    # clip proposals
    # filter proposals with either height or width < min_size
    # take top pre_nms_topN (objectiveness score) proposals before NMS
    # apply NMS
    # take top post_nms_topN proposals after NMS
    # return those top proposals
    # 
    # Parameters:
    #   rpn_cls_pred is the fore/background prediction
    #   rpn_box_reg is the box regression
    #   im_info = (img_width, img_height)
    #   feature_info = (feature's width, feature's height)
    #   anchor_info = (anchor nums, stride, aspect, scale)
    #   is_trian is the training state, default is True
"""


def get_region_proposal(rpn_cls_pred, rpn_box_reg, im_info, feature_info, anchor_info, cfg):
    # generate all anchors over the grid
    im_height, im_width = im_info
    H, W = feature_info
    A, stride, aspect, scale = anchor_info
    anchor_boxs = all_anchor_generator(feature_info, anchor_info)
    anchor_boxs = anchor_boxs.type_as(rpn_cls_pred)

    # apply delta to anchors to generate boxes (proposals)
    anchor_boxs = torch.reshape(anchor_boxs, (H * W * A, 4))
    rpn_box_reg = torch.reshape(rpn_box_reg, (H * W * A, 4))
    proposals = regformat_to_gtformat(rpn_box_reg, xyxy_to_xywh(anchor_boxs))
    # notice here we transfer the proposals format from xywh to xyxy
    proposals = xywh_to_xyxy(proposals)

    # clip proposals
    proposals = clip_img_boundary(proposals, im_width, im_height)
    keeps = filter_minsize_proposal(proposals, cfg['MIN_SIZE'])

    # filter proposals and cls_pred with either height or width < min_size
    proposals = proposals[keeps, :]
    rpn_cls_pred = torch.reshape(rpn_cls_pred, (H * W * A, 2))
    rpn_cls_pred = rpn_cls_pred[keeps, 1]

    # take top pre_nms_topN (objectiveness score) proposals before NMS
    # idx 0 is background, idx 1 is foreground
    pred, pred_idx = torch.sort(rpn_cls_pred, descending=True)
    pred_keep = pred_idx[:cfg['PROPOSAL_PRE_NMS']]
    proposals_keep = proposals[pred_keep, :]
    cls_pred_keep = rpn_cls_pred[pred_keep]

    # apply NMS
    nms_keep = nms(proposals_keep, cls_pred_keep, cfg['NMS_TH'])
    nms_proposal = proposals_keep[nms_keep, :]

    # take top post_nms_topN proposals after NMS
    final_proposal = nms_proposal[:cfg['PROPOSAL_POST_NMS'], :]

    # return those top proposals
    rois = final_proposal.new_zeros((final_proposal.size(0), 5))
    rois[:, 1:] = final_proposal
    return rois


if __name__ == '__main__':
    # get_region_proposal(None, None, (70,80), (5,6), (9,16,[0.5, 1, 2],[8, 16, 32]), is_train=True)
    pass
