import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from data import VOCDetection, FRDataLoader
from faster_rcnn import FASTER_RCNN
from configs import cfg, update_config
from utils.tools import warp_tqdm, setup_logger, get_scheduler, get_optimizer, AverageMeter
from utils.boxes import xyxy_to_xywh, xywh_to_xyxy, regformat_to_gtformat, clip_img_boundary

from utils.visualization import plot_bndbox
from utils.nms import nms
import torchvision.transforms as transforms


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


def train(model, train_loader, optimizer, scheduler, cfg, log):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    tqdm_train_loader = warp_tqdm(train_loader)

    rpn_cls_loss_meter = AverageMeter()
    rpn_reg_loss_meter = AverageMeter()
    rcnn_cls_loss_meter = AverageMeter()
    rcnn_reg_loss_meter = AverageMeter()
    total_loss_meter = AverageMeter()

    for i, (img, gt_box, gt_cls, flip) in enumerate(tqdm_train_loader):
        if not gt_box[0].shape[0]:
            continue

        img = img.to(device)
        gt_box = gt_box.to(device)
        gt_cls = gt_cls.to(device)
        batch_size, channels, h, w = img.shape
        for bid in range(batch_size):
            if flip[bid]:
                img[bid] = torch.flip(img[bid], [2])

        # show_img = transforms.ToPILImage()(img[0].cpu())
        # plot_bndbox(show_img, gt_box[0].cpu())

        rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss, rcnn_cls_prob, rcnn_rois = model(img, gt_box, gt_cls)
        loss = rpn_cls_loss + cfg['TRAIN']['RPN']['LAMBDA'] * rpn_reg_loss + \
               rcnn_cls_loss + cfg['TRAIN']['FAST_RCNN']['LAMBDA'] * rcnn_reg_loss

        rpn_cls_loss_meter.update(rpn_cls_loss.item())
        rpn_reg_loss_meter.update(rpn_reg_loss.item())
        rcnn_cls_loss_meter.update(rcnn_cls_loss.item())
        rcnn_reg_loss_meter.update(rcnn_reg_loss.item())
        total_loss_meter.update(loss.item())

        tqdm_train_loader.set_description('Loss => total_loss: {:.2f}, rpn_cls_loss: {:.2f}, rpn_reg_loss: {:.2f}, '
                                          'rcnn_cls_loss: {:.2f}, rcnn_reg_loss: {:.2f}'
                                          .format(total_loss_meter.avg, rpn_cls_loss_meter.avg, rpn_reg_loss_meter.avg,
                                                  rcnn_cls_loss_meter.avg, rcnn_reg_loss_meter.avg))
        log.info('Loss => total_loss: {:.2f}, rpn_cls_loss: {:.2f}, rpn_reg_loss: {:.2f}, '
                 'rcnn_cls_loss: {:.2f}, rcnn_reg_loss: {:.2f}'
                 .format(total_loss_meter.avg, rpn_cls_loss_meter.avg, rpn_reg_loss_meter.avg,
                         rcnn_cls_loss_meter.avg, rcnn_reg_loss_meter.avg))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        optimizer.step()

        if i != 0 and i % cfg['RESULT']['LOG_ITER'] == 0:
            rpn_cls_loss_meter.reset()
            rpn_reg_loss_meter.reset()
            rcnn_cls_loss_meter.reset()
            rcnn_reg_loss_meter.reset()
            total_loss_meter.reset()

    scheduler.step()


def test(model, test_set, test_loader, cfg, log):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    tqdm_test_loader = warp_tqdm(test_loader)
    num_classes = len(cfg['DATASET']['CLASSES'])
    num_images = len(test_set)

    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

    with torch.no_grad():
        for i, (img, ori_size, new_size, scale) in enumerate(tqdm_test_loader):
            img = img.to(device)
            nw, nh = new_size
            nw = nw.item()
            nh = nh.item()
            rcnn_cls_score, rcnn_reg_score, rcnn_cls_prob, rcnn_rois = model(img, [], [])

            bbox_normalize_means = torch.FloatTensor(cfg['TRAIN']['FAST_RCNN']['BBOX_NORMALIZE_MEANS']).\
                type_as(rcnn_reg_score)
            bbox_normalize_stds = torch.FloatTensor(cfg['TRAIN']['FAST_RCNN']['BBOX_NORMALIZE_STDS']).\
                type_as(rcnn_reg_score)

            roi = rcnn_rois[:, 1:]

            # test code
            # all_box_nums = rcnn_cls_prob.shape[0]
            for j in range(1, num_classes):
                prob = rcnn_cls_prob[:, j]
                reg = rcnn_reg_score[:, j*4:(j+1)*4]

                # Normalize regression target
                if cfg['TRAIN']['FAST_RCNN']['BBOX_NORMALIZE_TARGETS']:
                    reg = reg * bbox_normalize_stds.view(1, 4) + bbox_normalize_means.view(1, 4)
                bndbox = xywh_to_xyxy(regformat_to_gtformat(reg, xyxy_to_xywh(roi)))
                bndbox = clip_img_boundary(bndbox, nw, nh)
                picked = torch.where(prob > cfg['TEST']['FAST_RCNN']['PROB_TH'])[0]
                # picked = torch.where(prob > 0.05)[0]
                if picked.shape[0]:
                    nms_keep = nms(bndbox[picked], prob[picked], cfg['TEST']['FAST_RCNN']['NMS_TH'])
                    final_picked = picked[nms_keep]
                    # show_img = transforms.ToPILImage()(img[0].cpu())
                    # plot_bndbox(show_img, bndbox[final_picked].cpu(),
                    #             [cfg['DATASET']['CLASSES'][j] for _ in range(final_picked.shape[0])])

                    picked_prob = prob[final_picked][:cfg['TEST']['FAST_RCNN']['MAX_BOX_NUM']].unsqueeze(1)
                    # a = bndbox[final_picked]
                    picked_bndbox = bndbox[final_picked][:cfg['TEST']['FAST_RCNN']['MAX_BOX_NUM']] / scale.to(device)
                    picked_bndbox = picked_bndbox.type_as(picked_prob)

                    cls_dets = torch.cat((picked_bndbox, picked_prob), dim=1)
                    all_boxes[j][i] = cls_dets.cpu().numpy()
                else:
                    all_boxes[j][i] = empty_array
        test_set.evaluate_detections(all_boxes, 'output')


def main(cfg):
    # initial logger
    log = setup_logger(os.path.join(cfg['RESULT']['SAVE_PATH'], cfg['RESULT']['LOGGER']))
    log.info(cfg)

    # create basic model
    log.info("=> creating model with backbone '{}'".format(cfg['MODEL']['BACKBONE']))
    # model = FASTER_RCNN(cfg)
    model = FASTER_RCNN(cfg).cuda()
    # model = torch.load('faster_rcnn_model_epoch10.pth')

    # model = nn.DataParallel(model).cuda()

    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': cfg['SOLVER']['BASE_LR'] * (cfg['SOLVER']['DOUBLE_BIAS'] + 1),
                            'weight_decay': cfg['SOLVER']['BIAS_DECAY'] and cfg['SOLVER']['WEIGHT_DECAY'] or 0}]
            else:
                params += [{'params': [value], 'lr': cfg['SOLVER']['BASE_LR'],
                            'weight_decay': cfg['SOLVER']['WEIGHT_DECAY']}]

    optimizer = get_optimizer(params, cfg['SOLVER'])
    scheduler = get_scheduler(optimizer, cfg['SOLVER'])

    # cudnn.benchmark = True

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg['DATASET']['PIXEL_MEAN'],
                             std=cfg['DATASET']['PIXEL_STD']),
    ])

    if cfg['EVAL']:
        test_voc_dataset = VOCDetection(cfg, transform, is_train_set=False)
        rpn_test_loader = FRDataLoader(test_voc_dataset, False, cfg)
        test(model, test_voc_dataset, rpn_test_loader, cfg, log)
        return

    voc_dataset = VOCDetection(cfg, transform, is_train_set=True)
    rpn_loader = FRDataLoader(voc_dataset, True, cfg)

    for e in range(cfg['SOLVER']['START_EPOCH'], cfg['SOLVER']['END_EPOCH']):
        log.info('Epoch: [{}/{}]'.format(e+1, cfg['SOLVER']['END_EPOCH']))
        train(model, rpn_loader, optimizer, scheduler, cfg, log)
        torch.save(model, 'faster_rcnn_model_epoch' + str(e+1) + '.pth')


if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args)
    main(cfg)
