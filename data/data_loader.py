import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .dataset import VOCDetection


# faster rcnn data loader
class FRDataLoader(DataLoader):
    # TODO 添加一些sampling的方法
    def __init__(self, dataset, shuffle, cfg):
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=cfg['DATASET']['PIXEL_MEAN'],
        #                          std=cfg['DATASET']['PIXEL_STD']),
        # ])
        assert cfg['DATALOADER']['BATCH_SIZE'] == 1, 'Only support batch_size = 1'

        super().__init__(dataset, batch_size=cfg['DATALOADER']['BATCH_SIZE'], shuffle=shuffle,
                         num_workers=cfg['DATALOADER']['NUM_WORKERS'])
