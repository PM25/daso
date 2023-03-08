import math
import os

import numpy as np
import torch
from scipy import optimize

from lib.dataset.utils import get_data_config, get_imb_num


def get_target_dist(cfg, to_prob=False, device=None):
    data_cfg = get_data_config(cfg)

    num_l_head = data_cfg.NUM_LABELED_HEAD
    num_ul_head = data_cfg.NUM_UNLABELED_HEAD
    imb_factor_l = data_cfg.IMB_FACTOR_L
    imb_factor_ul = data_cfg.IMB_FACTOR_UL
    reverse_ul = cfg.DATASET.REVERSE_UL_DISTRIBUTION

    ul_samples_per_class = get_imb_num(
        num_ul_head,
        imb_factor_ul,
        num_classes=cfg.MODEL.NUM_CLASSES,
        reverse=reverse_ul,
        normalize=False
    ) if imb_factor_ul > 0 else [100000]  # stl10

    is_dist_equal = (imb_factor_l == imb_factor_ul) and (not reverse_ul)
    if not is_dist_equal:
        # load from pre-computed (estimated) unlabeled distribution
        estim_path = cfg.ALGORITHM.DARP.EST
        
        if reverse_ul:
            imb_factor_ul = 1 / imb_factor_ul
        
        if cfg.DATASET.NAME == "stl10":
            est_path = f"dist_estimation/{cfg.DATASET.NAME}_n{num_l_head}_l{imb_factor_l}_long_s{cfg.SEED}/BBSE_None_estimation.json"
        else:
            est_path = f"dist_estimation/{cfg.DATASET.NAME}_n{num_l_head}_m{num_ul_head}_l{imb_factor_l}_u{imb_factor_ul}_long_s{cfg.SEED}/BBSE_None_estimation.json"
        # p(y) based on the labeled examples seen during training
        from pathlib import Path
        import json
        assert Path(est_path).exists(), f"Can't Find the Distribution Estimation at {est_path}"
        with open(est_path, "r") as f:
            p_target = json.loads(f.read())
            p_target = torch.tensor(p_target["distribution"])
            ulb_dist = p_target * sum(ul_samples_per_class)
            ulb_dist_np = ulb_dist.cpu().numpy()        
        target_dist = sum(ul_samples_per_class) * ulb_dist_np / np.sum(ulb_dist_np)
        
        print("loaded estimated distribution from: {}".format(est_path))
        print("[BBSE None] The scaled distribution is as following:")
        print(target_dist)
    else:
        # assume labeled distribution equals unlabeled distribution
        target_dist = ul_samples_per_class
    target_dist = torch.Tensor(target_dist).float()  # cpu

    if to_prob:
        target_dist = target_dist / target_dist.sum()
    if device is not None:
        target_dist = target_dist.to(device)

    return target_dist
