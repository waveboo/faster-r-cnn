import torch
import collections
import torchvision.transforms as transforms


# targetSize is the img short side target
# maxSize is the img long side max
def resize_img(img, target_size, max_size):
    w = img.size[0]
    h = img.size[1]
    minDim = min(w, h)
    maxDim = max(w, h)
    scale = target_size / minDim
    if scale * maxDim > max_size:
        scale = max_size / maxDim
    nw = int(w * scale)
    nh = int(h * scale)
    transform = transforms.Compose([
        transforms.Resize([nh, nw]),
    ])
    img = transform(img)
    ori_size = (w, h)
    new_size = (nw, nh)
    return img, scale, ori_size, new_size


# transform bndbox to the target size
def trans_gt(target, n_size, scale, flip, classes, use_hard=False):
    nw, nh = n_size
    bndbox = []
    cls = []
    for obj in target['annotation']['object']:
        if not use_hard and int(obj['difficult']) == 1:
            continue
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
        cls.append(classes.index(obj['name']))
    if bndbox:
        return torch.stack(bndbox, dim=0), torch.Tensor(cls)
    else:
        return torch.Tensor([]). torch.Tensor([])


def parse_voc_xml(node):
    voc_dict = {}
    children = list(node)
    if children:
        def_dic = collections.defaultdict(list)
        for dc in map(parse_voc_xml, children):
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
