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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    tqdm_train_loader = warp_tqdm(train_loader)
    for i, (img, gt_box, gt_cls, flip) in enumerate(tqdm_train_loader):
        img = img.to(device)
        gt_box = gt_box.to(device)
        gt_cls = gt_cls.to(device)
        batch_size, channels, h, w = img.shape
        for bid in range(batch_size):
            if flip[bid]:
                img[bid] = torch.flip(img[bid], [2])

        rpn_cls_loss, rpn_reg_loss, rpn_rois, rcnn_cls_loss, rcnn_reg_loss, rcnn_cls_prob, rcnn_rois = \
            model(img, gt_box, gt_cls)
        loss = rpn_cls_loss + cfg['TRAIN']['RPN']['LAMBDA'] * rpn_reg_loss + \
               rcnn_cls_loss + cfg['TRAIN']['FAST_RCNN']['LAMBDA'] * rcnn_reg_loss

        tqdm_train_loader.set_description('Loss => total_loss: {:.2f}, rpn_cls_loss: {:.2f}, rpn_reg_loss: {:.2f}, '
                                          'rcnn_cls_loss: {:.2f}, rcnn_reg_loss: {:.2f}'
                                          .format(loss.item(), rpn_cls_loss.item(), rpn_reg_loss.item(),
                                                  rcnn_cls_loss.item(), rcnn_reg_loss.item()))
        log.info('Loss => total_loss: {:.2f}, rpn_cls_loss: {:.2f}, rpn_reg_loss: {:.2f}, '
                 'rcnn_cls_loss: {:.2f}, rcnn_reg_loss: {:.2f}'
                 .format(loss.item(), rpn_cls_loss.item(), rpn_reg_loss.item(),
                         rcnn_cls_loss.item(), rcnn_reg_loss.item()))

        # rpn_cls_loss, rpn_reg_loss, rpn_rois = model(img, gt_box, gt_cls)
        # loss = rpn_cls_loss + cfg['TRAIN']['RPN']['LAMBDA'] * rpn_reg_loss
        # tqdm_train_loader.set_description('Loss => total_loss: {:.2f}, rpn_cls_loss: {:.2f}, rpn_reg_loss: {:.2f}'
        #                                   .format(loss.item(), rpn_cls_loss.item(), rpn_reg_loss.item()))
        # log.info('Loss => total_loss: {:.2f}, rpn_cls_loss: {:.2f}, rpn_reg_loss: {:.2f}'
        #          .format(loss.item(), rpn_cls_loss.item(), rpn_reg_loss.item()))

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
    # model = FASTER_RCNN(cfg)
    # model = FASTER_RCNN(cfg).cuda()
    model = torch.load('faster_rcnn_model.pth')

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
    torch.save(model, 'faster_rcnn_model.pth')


if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args)
    main(cfg)
