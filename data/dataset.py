import os
import collections
import xml.etree.ElementTree as ET
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from utils.visualization import plot_bndbox


# from https://pytorch.org/docs/stable/_modules/torchvision/datasets/voc.html#VOCDetection
class VOCDetection(Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, `train`, `test`, `val`, or `trainval`
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, root, cfg, image_set='trainval', transform=None, random_horizontal_flip=True):
        # super(VOCDetection, self).__init__(root, image_set, transform)
        valid_sets = ["train", "trainval", "val", "test"]
        assert image_set in valid_sets

        if not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted.')

        splits_dir = os.path.join(root, 'ImageSets/Segmentation')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        image_dir = os.path.join(root, 'JPEGImages')
        annotation_dir = os.path.join(root, 'Annotations')

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations))

        self.transform = transform
        self.random_horizontal_flip = random_horizontal_flip

        self.classes = cfg['DATASET']['CLASSES']
        self.target_size = cfg['DATASET']['TARGET_SIZE']
        self.max_size = cfg['DATASET']['MAX_SIZE']

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        if not self.random_horizontal_flip:
            assert index < len(self.images)
        else:
            assert index < 2 * len(self.images)

        if index >= len(self.images):
            flip = True
        else:
            flip = False

        img = Image.open(self.images[index%len(self.images)]).convert('RGB')
        target = self.parse_voc_xml(ET.parse(self.annotations[index%len(self.images)]).getroot())

        img, bndbox, cls = self.trans_img_gt(img, target, flip)

        if self.transform is not None:
            img = self.transform(img)

        return img, bndbox, cls, flip

    def __len__(self):
        if self.random_horizontal_flip:
            return 2 * len(self.images)
        else:
            return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

    # transform image rescaling and bndbox
    def trans_img_gt(self, img, target, flip):
        w = img.size[0]
        h = img.size[1]
        minDim = min(w, h)
        maxDim = max(w, h)
        scale = self.target_size / minDim
        if scale * maxDim > self.max_size:
            scale = self.max_size / maxDim
        nw = int(w * scale)
        nh = int(h * scale)
        transform = transforms.Compose([
            transforms.Resize([nh, nw]),
        ])
        img = transform(img)

        bndbox = []
        cls = []
        for obj in target['annotation']['object']:
            # make the position start from 0
            px1 = int(obj['bndbox']['xmin'])-1
            py1 = int(obj['bndbox']['ymin'])-1
            px2 = int(obj['bndbox']['xmax'])-1
            py2 = int(obj['bndbox']['ymax'])-1
            if flip:
                point = torch.Tensor([nw-int((px2+1)*scale), int(py1*scale),
                                      nw-int((px1+1)*scale), int(py2*scale)])
            else:
                point = torch.Tensor([int(px1*scale), int(py1*scale),
                                      int(px2*scale), int(py2*scale)])
            bndbox.append(point)
            cls.append(self.classes.index(obj['name']))
        return img, torch.stack(bndbox, dim=0), torch.Tensor(cls)


if __name__ == '__main__':
    pass
    # transform = transforms.Compose([
    #     # transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225]),
    # ])
    # voc = VOCDetection('/Users/SweetBelle/Documents/datasets/VOCdevkit/VOC2012', 'trainval',
    #                    transform, random_horizontal_flip=True)
    # # print(len(voc))
    # # print(voc[5825])
    #
    # # for i in range(2913, 2913+5):
    # for i in range(5):
    #     img, bndbox, c, flip = voc[i]
    #     obj_points = []
    #     for j in bndbox:
    #         obj_points.append(j)
    #     if flip:
    #         img = transforms.ToPILImage()(img).transpose(Image.FLIP_LEFT_RIGHT)
    #     else:
    #         img = transforms.ToPILImage()(img)
    #     plot_bndbox(img, obj_points)

