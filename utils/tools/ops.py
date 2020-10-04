import os
import tqdm
import torch
import shutil
import pickle


def warp_tqdm(data_loader):
    tqdm_loader = tqdm.tqdm(data_loader, total=len(data_loader))
    return tqdm_loader


# def save_checkpoint(state, is_best, cfg):
#     torch.save(state, os.path.join(cfg['RESULT']['SAVE_PATH'], cfg['RESULT']['FEATURE_EXTRACTOR_LAST_MODEL']))
#     if is_best:
#         shutil.copyfile(os.path.join(cfg['RESULT']['SAVE_PATH'], cfg['RESULT']['FEATURE_EXTRACTOR_LAST_MODEL']),
#                         os.path.join(cfg['RESULT']['SAVE_PATH'], cfg['RESULT']['FEATURE_EXTRACTOR_BEST_MODEL']))
#
#
# def load_checkpoint(model, cfg, type='best'):
#     if type == 'best':
#         checkpoint = torch.load(os.path.join(cfg['RESULT']['SAVE_PATH'],
#                                              cfg['RESULT']['FEATURE_EXTRACTOR_BEST_MODEL']))
#     elif type == 'last':
#         checkpoint = torch.load(os.path.join(cfg['RESULT']['SAVE_PATH'],
#                                              cfg['RESULT']['FEATURE_EXTRACTOR_LAST_MODEL']))
#     else:
#         assert False, 'type should be in [best, or last], but got {}'.format(type)
#     model.load_state_dict(checkpoint['state_dict'])
#
#
# def save_pickle(file, data):
#     with open(file, 'wb') as f:
#         pickle.dump(data, f)
#         f.close()
#
#
# def load_pickle(file):
#     with open(file, 'rb') as f:
#         res = pickle.load(f)
#         f.close()
#     return res
