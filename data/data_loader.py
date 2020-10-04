import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .dataset import VOCDetection


# faster rcnn data loader
class FRDataLoader(DataLoader):
    # TODO 添加一些sampling的方法
    def __init__(self, cfg, is_train=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg['DATASET']['PIXEL_MEAN'],
                                 std=cfg['DATASET']['PIXEL_STD']),
        ])
        if is_train:
            dataset = VOCDetection(cfg['DATASET']['ROOT_PATH'], cfg, 'trainval', transform,
                                   random_horizontal_flip=cfg['DATASET']['HORIZONTAL_FLIP'])
        else:
            dataset = VOCDetection(cfg['DATASET']['ROOT_PATH'], cfg, 'test', transform, random_horizontal_flip=False)
        super().__init__(dataset, batch_size=cfg['DATALOADER']['BATCH_SIZE'], shuffle=True,
                         num_workers=cfg['DATALOADER']['NUM_WORKERS'])
