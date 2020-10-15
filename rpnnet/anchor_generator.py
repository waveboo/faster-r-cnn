import torch
# from data.dataset import VOCDetection
# from configs import scale, aspect


# Ross's is
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360
#
# My is
#    anchors =
#
#       -84   -40    99    55
#      -176   -88   191   103
#      -360  -184   375   199
#       -56   -56    71    71
#      -120  -120   135   135
#      -248  -248   263   263
#       -36   -80    51    95
#       -80  -168    95   183
#      -168  -344   183   359
def base_anchor_generator(stride=16, aspect=[0.5, 1, 2], scale=[8, 16, 32]):
    base_anchor = torch.Tensor([0, 0, stride-1, stride-1])
    anchors = _scale_aspect_anchors(base_anchor, torch.Tensor(aspect), scale)
    return anchors


def _scale_aspect_anchors(base_anchor, aspect, scale):
    w = base_anchor[2] - base_anchor[0] + 1
    h = base_anchor[3] - base_anchor[1] + 1
    xc = (base_anchor[2] + base_anchor[0]) / 2
    yc = (base_anchor[3] + base_anchor[1]) / 2
    size = w*h
    nw = torch.round(torch.sqrt(size / aspect))
    nh = torch.round(nw * aspect)
    return torch.cat([_mk_anchor(xc, yc, nw*s, nh*s) for s in scale], dim=0)


def _mk_anchor(xc, yc, w, h):
    return torch.stack((xc-(w-1)/2, yc-(h-1)/2, xc+(w-1)/2, yc+(h-1)/2), dim=1)


# generate all anchors for the given feature grid
def all_anchor_generator(feature_info, anchor_info):
    H, W = feature_info
    A, stride, aspect, scale = anchor_info
    x_step = torch.arange(W) * stride
    y_step = torch.arange(H) * stride
    xs, ys = torch.meshgrid(x_step, y_step)
    xys = torch.stack([xs.T, ys.T], dim=-1)
    shift = torch.stack([xys[:, :, 0], xys[:, :, 1], xys[:, :, 0], xys[:, :, 1]], dim=-1)
    shift = shift.unsqueeze(2)
    shift = shift.expand(-1, -1, A, -1)
    anchor_boxs = shift + base_anchor_generator(stride, aspect, scale)
    return anchor_boxs


if __name__ == '__main__':
    # print(anchor_generator())
    pass




