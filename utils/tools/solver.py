import torch
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR

# cfg_solver = cfg['FEATURE_EXTRACTOR']['SOLVER']


def get_optimizer(module, cfg_solver):
    OPTIMIZER = {'SGD': torch.optim.SGD(module.parameters(), lr=cfg_solver['BASE_LR'],
                                        momentum=0.9, weight_decay=cfg_solver['WEIGHT_DECAY']),
                 'Adam': torch.optim.Adam(module.parameters(), lr=cfg_solver['BASE_LR'])}
    return OPTIMIZER[cfg_solver['OPTIMIZER']]


def get_scheduler(batches, optimiter, cfg_solver):
    """
    cosine will change learning rate every iteration, others change learning rate every epoch
    :param batches: the number of iterations in each epochs
    :return: scheduler
    """
    SCHEDULER = {'StepLR': StepLR(optimiter, cfg_solver['STEP_SIZE'],
                                  cfg_solver['GAMMA']),
                 'MultiStepLR': MultiStepLR(optimiter, milestones=cfg_solver['STEPS'],
                                            gamma=cfg_solver['GAMMA']),
                 'CosineAnnealingLR': CosineAnnealingLR(optimiter,
                                                        batches * cfg_solver['EPOCHS'],
                                                        eta_min=1e-9)}
    return SCHEDULER[cfg_solver['LR_SCHEDULER_NAME']]
