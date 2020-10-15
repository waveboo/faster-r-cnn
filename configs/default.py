# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()


_C.DATASET = CN()
_C.DATASET.TRAIN = CN()
_C.DATASET.TRAIN.ROOT_PATH = ('/home/lb/datasets/VOCdevkit/VOC2012', )
_C.DATASET.TRAIN.SPLIT_TYPE = ('trainval', )
_C.DATASET.TRAIN.HORIZONTAL_FLIP = True
_C.DATASET.TRAIN.USE_HARD_EXAM = False

_C.DATASET.TEST = CN()
_C.DATASET.TEST.ROOT_PATH = ('/home/lb/datasets/VOCdevkit/VOC2007', )
_C.DATASET.TEST.SPLIT_TYPE = ('test', )
_C.DATASET.TEST.USE_07_METRIC = True

_C.DATASET.TARGET_SIZE = 600
_C.DATASET.MAX_SIZE = 1000
_C.DATASET.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.DATASET.PIXEL_STD = [0.229, 0.224, 0.225]
_C.DATASET.CLASSES = ('__background__', 'aeroplane', 'bicycle', 'bird',
                      'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                      'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                      'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


_C.MODEL = CN()
_C.MODEL.BACKBONE = 'resnet50'
_C.MODEL.N_FEATURES = 1024
_C.MODEL.POOLING_SIZE = 7
_C.MODEL.PRE_TRAIN = True


_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = "SGD"
_C.SOLVER.START_EPOCH = 0
_C.SOLVER.END_EPOCH = 10
_C.SOLVER.BASE_LR = 1e-3
_C.SOLVER.LR_SCHEDULER_NAME = "MultiStepLR"
_C.SOLVER.GAMMA = 0.1
# for StepLR
_C.SOLVER.STEP_SIZE = 6
# for MultiStepLR
_C.SOLVER.STEPS = [6, 8]
_C.SOLVER.WEIGHT_DECAY = 1e-4
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.DOUBLE_BIAS = True
_C.SOLVER.BIAS_DECAY = False


_C.DATALOADER = CN()
# only support batch_size=1
_C.DATALOADER.BATCH_SIZE = 1
_C.DATALOADER.NUM_WORKERS = 1


_C.ANCHORS = CN()
_C.ANCHORS.STRIDE = 16
_C.ANCHORS.SCALE = [8, 16, 32]
_C.ANCHORS.ASPECT = [0.5, 1, 2]


_C.TRAIN = CN()

_C.TRAIN.RPN = CN()
_C.TRAIN.RPN.MIN_SIZE = 16
_C.TRAIN.RPN.PROPOSAL_PRE_NMS = 12000
_C.TRAIN.RPN.PROPOSAL_POST_NMS = 2000
_C.TRAIN.RPN.NMS_TH = 0.7
_C.TRAIN.RPN.POS_ANCHOR_TH = 0.7
_C.TRAIN.RPN.NEG_ANCHOR_TH = 0.3
_C.TRAIN.RPN.ANCHORS_PER_IMG = 256
_C.TRAIN.RPN.ANCHORS_POS_PER_IMG = 128
_C.TRAIN.RPN.CLS_WEIGHT_DETLA = 1.0
_C.TRAIN.RPN.REG_WEIGHT_DETLA = 1.0
_C.TRAIN.RPN.LAMBDA = 1

_C.TRAIN.FAST_RCNN = CN()
_C.TRAIN.FAST_RCNN.ROIS_PER_IMG = 64
_C.TRAIN.FAST_RCNN.POS_FRACTION = 0.25
_C.TRAIN.FAST_RCNN.POS_ROI_TH = 0.5
_C.TRAIN.FAST_RCNN.NEG_ROI_TH = 0.1
_C.TRAIN.FAST_RCNN.REG_WEIGHT_DETLA = 1.0
_C.TRAIN.FAST_RCNN.BBOX_NORMALIZE_TARGETS = True
_C.TRAIN.FAST_RCNN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
_C.TRAIN.FAST_RCNN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
_C.TRAIN.FAST_RCNN.LAMBDA = 1

_C.TEST = CN()

_C.TEST.RPN = CN()
_C.TEST.RPN.MIN_SIZE = 16
_C.TEST.RPN.PROPOSAL_PRE_NMS = 6000
_C.TEST.RPN.PROPOSAL_POST_NMS = 300
_C.TEST.RPN.NMS_TH = 0.7

_C.TEST.FAST_RCNN = CN()
_C.TEST.FAST_RCNN.NMS_TH = 0.3
_C.TEST.FAST_RCNN.PROB_TH = 0.0
_C.TEST.FAST_RCNN.MAX_BOX_NUM = 100

_C.EVAL = False


_C.RESULT = CN()
_C.RESULT.LOG_ITER = 100
_C.RESULT.SAVE_PATH = ''
_C.RESULT.LOGGER = ''
# _C.RESULT.LOG_FREQ = 10
# _C.RESULT.FEATURE_EXTRACTOR_LAST_MODEL = ''
# _C.RESULT.FEATURE_EXTRACTOR_BEST_MODEL = ''
# _C.RESULT.FEATURE_EXTRACTOR_PATH = ''


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)

    if args.eval_only:
        cfg.EVAL = True

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

