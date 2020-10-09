import torch


def iou_calcu(rect1, rect2):
    # rect1 is N1 * 4, in (xmin, ymin, xmax, ymax) format
    # rect2 is N2 * 4, in (xmin, ymin, xmax, ymax) format
    # return iou is the intersection over union (IoU) between
    #                       rect1 and rect2 is shape N1 * N2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    N1 = rect1.shape[0]
    N2 = rect2.shape[0]

    # the minth max x and y
    x_inter_minmax = torch.max(rect1[:, 0].view(N1, 1), rect2[:, 0].view(1, N2))
    y_inter_minmax = torch.max(rect1[:, 1].view(N1, 1), rect2[:, 1].view(1, N2))
    # the maxth min x and y
    x_inter_maxmin = torch.min(rect1[:, 2].view(N1, 1), rect2[:, 2].view(1, N2))
    y_inter_maxmin = torch.min(rect1[:, 3].view(N1, 1), rect2[:, 3].view(1, N2))

    # the intersection part
    inter_w = x_inter_maxmin - x_inter_minmax
    inter_h = y_inter_maxmin - y_inter_minmax

    # compare with the zero tensor(the 2nd param must be tensor! otherwise it will represent the dim)
    inter_w = torch.max(inter_w, torch.Tensor([0]).to(device))
    inter_h = torch.max(inter_h, torch.Tensor([0]).to(device))

    inter = inter_w * inter_h

    # the union part
    union = (rect1[:, 2].view(N1, 1) - rect1[:, 0].view(N1, 1)) * (rect1[:, 3].view(N1, 1) - rect1[:, 1].view(N1, 1)) \
            + (rect2[:, 2].view(1, N2) - rect2[:, 0].view(1, N2)) * (rect2[:, 3].view(1, N2) - rect2[:, 1].view(1, N2))\
            - inter

    return inter / union


def xyxy_to_xywh(rect):
    # rect is N * 4, in (xmin, ymin, xmax, ymax) format
    # return ans xywh is N * 4, in (xcen, ycen, width, height) format

    xcen = (rect[:, 0] + rect[:, 2]) / 2
    ycen = (rect[:, 1] + rect[:, 3]) / 2
    w = rect[:, 2] - rect[:, 0] + 1
    h = rect[:, 3] - rect[:, 1] + 1
    return torch.stack([xcen, ycen, w, h], dim=-1)


def xywh_to_xyxy(rect):
    # rect is N * 4, in (xcen, ycen, width, height) format
    # return ans xywh is N * 4, in (xmin, ymin, xmax, ymax) format

    xmin = rect[:, 0] - (rect[:, 2] - 1) / 2
    ymin = rect[:, 1] - (rect[:, 3] - 1) / 2
    xmax = rect[:, 0] + (rect[:, 2] - 1) / 2
    ymax = rect[:, 1] + (rect[:, 3] - 1) / 2
    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)


def gtformat_to_regformat(gt_boxes, anchor_boxes):
    # gt_boxes is ground_truth format box, in (x, y, w, h) format, N * 4
    # anchor_boxes is anchor boxes, in (xa, ya, wa, ha) format, N * 4
    # return ans reg_boxes is N * 4, in (tx, ty, tw, th) format

    tx = (gt_boxes[:, 0] - anchor_boxes[:, 0]) / anchor_boxes[:, 2]
    ty = (gt_boxes[:, 1] - anchor_boxes[:, 1]) / anchor_boxes[:, 3]
    tw = torch.log(gt_boxes[:, 2] / anchor_boxes[:, 2])
    th = torch.log(gt_boxes[:, 3] / anchor_boxes[:, 3])
    return torch.stack([tx, ty, tw, th], dim=-1)


def regformat_to_gtformat(reg_boxes, anchor_boxes):
    # reg_boxes is regression format box, in (tx, ty, tw, th) format, N * 4
    # anchor_boxes is anchor boxes, in (xa, ya, wa, ha) format, N * 4
    # return ans gt_boxes is N * 4, in (x, y, w, h) format

    x = reg_boxes[:, 0] * anchor_boxes[:, 2] + anchor_boxes[:, 0]
    y = reg_boxes[:, 1] * anchor_boxes[:, 3] + anchor_boxes[:, 1]
    w = torch.exp(reg_boxes[:, 2]) * anchor_boxes[:, 2]
    h = torch.exp(reg_boxes[:, 3]) * anchor_boxes[:, 3]
    return torch.stack([x, y, w, h], dim=-1)


def clip_img_boundary(boxes, im_width, im_height):
    # boxes is the proposals
    # clip all proposals into [0, im_width, 0, im_height]

    boxes[:, 0::2].clamp_(0, im_width - 1)
    boxes[:, 1::2].clamp_(0, im_height - 1)
    return boxes


def filter_minsize_proposal(boxes, min_rpn_size):
    # boxes is the proposals
    # all proposals' size less equal than min_rpn_size will be ignored
    # return the keeps ones state  (a list of True or False)

    keeps = ((boxes[:, 2] - boxes[:, 0] + 1) >= min_rpn_size) & \
           ((boxes[:, 3] - boxes[:, 1] + 1) >= min_rpn_size)
    return keeps


def filter_boundary_anchor(boxes, im_width, im_height):
    # boxes is the anchors
    # all anchors' size cross the img boundary will be ignored
    # return the keeps ones state  (a list of True or False)
    keeps = (boxes[:, 0] >= 0) & (boxes[:, 2] < im_width) & \
            (boxes[:, 1] >= 0) & (boxes[:, 3] < im_height)
    return keeps


if __name__ == '__main__':
    pass
    # iou = iou_calcu(torch.Tensor([[1,1,5,5],[2,2,5,5]]),torch.Tensor([[4,4,6,6],[10,10,18,18],[2,3,5,6]]))
    # print(iou)
    # xywh = xyxy_to_xywh(torch.Tensor([[1,1,5,5],[2,2,5,5]]))
    # print(xywh)
    # xyxy = xywh_to_xyxy(xywh)
    # print(xyxy)
    # clip_boxes(torch.Tensor([[-1,-1,5,5],[2,2,5,5]]), [10, 10])

    # keeps = filter_minsize_proposal(torch.Tensor([[-1,-1,10,10],[2,2,5,5]]), 10)
    # print(keeps)