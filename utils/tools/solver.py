import torch
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR

# cfg_solver = cfg['FEATURE_EXTRACTOR']['SOLVER']


def get_optimizer(params, cfg_solver):
    OPTIMIZER = {'SGD': torch.optim.SGD(params, lr=cfg_solver['BASE_LR'],
                                        momentum=cfg_solver['MOMENTUM'], weight_decay=cfg_solver['WEIGHT_DECAY']),
                 'Adam': torch.optim.Adam(params, lr=cfg_solver['BASE_LR'])}
    return OPTIMIZER[cfg_solver['OPTIMIZER']]


def get_scheduler(optimiter, cfg_solver):
    """
    :return: scheduler
    """
    SCHEDULER = {'StepLR': StepLR(optimiter, cfg_solver['STEP_SIZE'],
                                  cfg_solver['GAMMA']),
                 'MultiStepLR': MultiStepLR(optimiter, milestones=cfg_solver['STEPS'],
                                            gamma=cfg_solver['GAMMA'])}
    return SCHEDULER[cfg_solver['LR_SCHEDULER_NAME']]
