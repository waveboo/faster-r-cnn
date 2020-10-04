import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from data import FRDataLoader
from faster_rcnn import FASTER_RCNN
from configs import cfg, update_config
from utils.tools import warp_tqdm, setup_logger, get_scheduler, get_optimizer


def parse_args():
    parser = argparse.ArgumentParser(description='Faster R-CNN Configs')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--eval-only',
                        action='store_true',
                        help='do evaluation only')
    args = parser.parse_args()
    return args


def train(model, train_loader, optimizer, cfg, log):
    model.train()
    tqdm_train_loader = warp_tqdm(train_loader)
    for i, (img, gt_box, gt_cls, flip) in enumerate(tqdm_train_loader):
        batch_size, channels, h, w = img.shape
        for bid in range(batch_size):
            if flip[bid]:
                img[bid] = torch.flip(img[bid], [2])

        cls_loss, reg_loss, rois = model(img, gt_box, gt_cls)
        loss = cls_loss + cfg['TRAIN']['LAMBDA'] * reg_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main(cfg):
    # initial logger
    log = setup_logger(os.path.join(cfg['RESULT']['SAVE_PATH'], cfg['RESULT']['LOGGER']))
    log.info(cfg)

    # create basic model
    log.info("=> creating model with backbone '{}'".format(cfg['MODEL']['BACKBONE']))
    model = FASTER_RCNN(cfg)
    # model = nn.DataParallel(model).cuda()

    optimizer = get_optimizer(model, cfg['SOLVER'])

    # cudnn.benchmark = True

    if cfg['EVAL']:
        # do eval
        return

    rpn_loader = FRDataLoader(cfg, is_train=True)

    for e in range(cfg['SOLVER']['START_EPOCH'], cfg['SOLVER']['END_EPOCH']):
        log.info('Epoch: [{}/{}]'.format(e+1, cfg['SOLVER']['END_EPOCH']))
        train(model, rpn_loader, optimizer, cfg, log)


if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args)
    main(cfg)
