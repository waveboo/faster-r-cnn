import torch
from utils.boxes import iou_calcu


# from https://github.com/tztztztztz/yolov2.pytorch/blob/master/yolo_eval.py#yolo_nms
def nms(boxes, scores, threshold):
    """
    Apply Non-Maximum-Suppression on boxes according to their scores

    Arguments:
    boxes -- tensor of shape (N, 4) (x1, y1, x2, y2)
    scores -- tensor of shape (N) confidence
    threshold -- float. NMS threshold

    Returns:
    keep -- tensor of shape (None), index of boxes which should be retain.
    """

    score_sort_index = torch.sort(scores, dim=0, descending=True)[1]

    keep = []

    while score_sort_index.numel() > 0:

        i = score_sort_index[0]
        keep.append(i)

        if score_sort_index.numel() == 1:
            break

        cur_box = boxes[score_sort_index[0], :].view(-1, 4)
        res_box = boxes[score_sort_index[1:], :].view(-1, 4)

        ious = iou_calcu(cur_box, res_box).view(-1)

        inds = torch.nonzero(ious < threshold).squeeze()

        # +1 because the 0th is the current box
        score_sort_index = score_sort_index[inds + 1].view(-1)

    return torch.LongTensor(keep)


if __name__ == '__main__':
    pass