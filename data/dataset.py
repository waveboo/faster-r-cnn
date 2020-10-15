import os
import uuid
import pickle
import xml.etree.ElementTree as ET
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from utils.visualization import plot_bndbox

from data.voc_eval import voc_eval
from data.data_ops import resize_img, trans_gt, parse_voc_xml


# from https://pytorch.org/docs/stable/_modules/torchvision/datasets/voc.html#VOCDetection
class VOCDetection(Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, `train`, `test`, `val`, or `trainval`
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, cfg, transform=None, is_train_set=True):
        valid_sets = ["train", "trainval", "val", "test"]
        key = 'TRAIN' if is_train_set else 'TEST'
        root_paths = cfg['DATASET'][key]['ROOT_PATH']
        split_sets = cfg['DATASET'][key]['SPLIT_TYPE']
        assert len(root_paths) == len(split_sets), 'root_path should be same dim with split_set'

        if key == 'TEST':
            assert len(root_paths) == 1, 'Only support test one dataset'

        self.images = []
        self.annotations = []
        self.image_index = []

        dirs = len(root_paths)
        for i in range(dirs):
            root = root_paths[i]
            if not os.path.isdir(root):
                raise RuntimeError('Dataset {:s} not found or corrupted.'.format(root))
            split = split_sets[i]
            assert split in valid_sets, 'split must be train, val, trainval or test'

            splits_dir = os.path.join(root, 'ImageSets', 'Main')
            split_f = os.path.join(splits_dir, split + '.txt')

            with open(os.path.join(split_f), "r") as f:
                file_names = [x.strip() for x in f.readlines()]

            # only for test
            self.image_index = file_names

            image_dir = os.path.join(root, 'JPEGImages')
            annotation_dir = os.path.join(root, 'Annotations')

            images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
            annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
            assert (len(images) == len(annotations))

            self.images.extend(images)
            self.annotations.extend(annotations)

        self.transform = transform
        self.is_train = is_train_set
        self.cfg = cfg

        self.classes = cfg['DATASET']['CLASSES']
        self.target_size = cfg['DATASET']['TARGET_SIZE']
        self.max_size = cfg['DATASET']['MAX_SIZE']

    def __getitem__(self, index):
        img = Image.open(self.images[index % len(self.images)]).convert('RGB')
        img, scale, ori_size, new_size = resize_img(img, self.target_size, self.max_size)
        if self.transform is not None:
            img = self.transform(img)
        target = parse_voc_xml(ET.parse(self.annotations[index % len(self.images)]).getroot())

        if self.is_train:
            if self.cfg['DATASET']['TRAIN']['HORIZONTAL_FLIP'] and index >= len(self.images):
                flip = True
            else:
                flip = False
            bndbox, cls = trans_gt(target, new_size, scale, flip,
                                   self.classes, use_hard=self.cfg['DATASET']['TRAIN']['USE_HARD_EXAM'])
            return img, bndbox, cls, flip
        else:
            return img, ori_size, new_size, scale

    def __len__(self):
        if self.is_train:
            if self.cfg['DATASET']['TRAIN']['HORIZONTAL_FLIP']:
                return 2 * len(self.images)
        return len(self.images)

    def evaluate_detections(self, all_boxes, output_dir=None):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)

    def _get_voc_results_file_template(self):
        # VOCdevkit/VOC2007/results/Main/<comp_id>_det_test_aeroplane.txt
        filename = 'comp4_det_' + \
                   self.cfg['DATASET']['TEST']['SPLIT_TYPE'][0] + '_{:s}.txt'
        filedir = os.path.join(self.cfg['DATASET']['TEST']['ROOT_PATH'][0], 'results', 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        root = self.cfg['DATASET']['TEST']['ROOT_PATH'][0]
        annopath = os.path.join(root, 'Annotations', '{:s}.xml')
        imagesetfile = os.path.join(root, 'ImageSets', 'Main',
                                    self.cfg['DATASET']['TEST']['SPLIT_TYPE'][0] + '.txt')
        cachedir = os.path.join(root, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = self.cfg['DATASET']['TEST']['USE_07_METRIC']
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, annopath, imagesetfile, cls, cachedir,
                                     ovthresh=0.5, use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')


if __name__ == '__main__':
    pass