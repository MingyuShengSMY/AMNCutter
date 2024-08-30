import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random


def normalize_01_array(array: np.ndarray, axis=None):
    if axis is not None and not isinstance(axis, int):
        axis = tuple(axis)
    array_min = array.min(axis=axis, keepdims=axis is not None)
    array_max = array.max(axis=axis, keepdims=axis is not None)
    delta = array_max - array_min
    if delta.size > 1:
        delta[delta == 0] = 1e-10
    else:
        if delta == 0:
            delta = 1e-10
    result = (array - array_min) / delta
    return result


def normalize_01_tensor(array: torch.Tensor, axis=None):
    if axis is not None and not isinstance(axis, int):
        axis = list(axis)
    array_min = array.amin(dim=axis, keepdim=axis is not None)
    array_max = array.amax(dim=axis, keepdim=axis is not None)
    delta = array_max - array_min
    if delta.nelement() > 1:
        delta[delta == 0] = 1e-10
    else:
        if delta == 0:
            delta = 1e-10
    result = (array - array_min) / delta
    return result


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PL_SEED_WORKERS"] = f"1"


def load_config_file(config_file, save_path_name=True):
    if save_path_name:
        print(f"Loading Config File: '{config_file}'")
    config = json.load(open(config_file))
    if save_path_name:
        config['config_file_path'] = config_file
        config['config_file_name'] = ".".join(os.path.split(config_file)[1].split(".")[:-1])
    return config


def count_parameters(model: torch.nn.Module, train_param=False):
    return int(sum(p.numel() for p in model.parameters() if not train_param or p.requires_grad))


def resize_label_map(label_map: torch.Tensor, h_w):
    label_map = label_map.squeeze().long()
    # one_hot_map = F.one_hot(label_map).float().permute(2, 0, 1).unsqueeze(0)
    # one_hot_map_resized = F.interpolate(one_hot_map, size=h_w, mode="bicubic")
    # one_hot_map_resized = F.interpolate(one_hot_map, size=h_w, mode="bilinear")
    # new_label_map = torch.argmax(one_hot_map_resized, dim=1, keepdim=True).long().squeeze()

    new_label_map = TF.resize(label_map.unsqueeze(0), size=h_w, interpolation=TF.InterpolationMode.NEAREST_EXACT).squeeze()
    return new_label_map





