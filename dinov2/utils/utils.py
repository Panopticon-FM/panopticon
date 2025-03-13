# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import random
import subprocess
from urllib.parse import urlparse

import numpy as np
import torch
from torch import nn
import re

logger = logging.getLogger("dinov2")


def load_pretrained_weights(model, pretrained_weights, checkpoint_key='model', ):
    full_state_dict = {}

    for cfg in pretrained_weights:
        ckpt_path = cfg['path']
        logger.info(f"[CHECKPOINT] Processing pretrained weights from {ckpt_path}")
        if urlparse(ckpt_path).scheme:  # If it looks like an URL
            state_dict = torch.hub.load_state_dict_from_url(ckpt_path, map_location="cpu")
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")

        if 'checkpoint_key' in cfg:
            checkpoint_key = cfg['checkpoint_key']
            logger.info(f"Using checkpoint key: {checkpoint_key}")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            logger.info(f"[CHECKPOINT] Take key '{checkpoint_key}' in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        else:
            logger.error(f"[CHECKPOINT] checkpoint_key '{checkpoint_key}' not found in provided checkpoint dict")

        if 'include' in cfg:
            patterns = cfg.include.split(',')
            keys_to_include = []
            for pattern in patterns:
                keys_to_include += [k for k in state_dict.keys() if re.search(pattern, k)]
                # logger.info(f'[CHECKPOINT] Include keys matching pattern "{pattern}": {keys_to_include}')
            state_dict = {k: v for k, v in state_dict.items() if k in keys_to_include}
            # logger.info(f'\n[CHECKPOINT] After Included keys matching patterns: {state_dict.keys()}')

        if 'exclude' in cfg:
            patterns = cfg.exclude.split(',')
            keys_to_drop = []
            for pattern in patterns:
                keys_to_drop += [k for k in state_dict.keys() if re.search(pattern, k)]
            state_dict = {k: v for k, v in state_dict.items() if k not in keys_to_drop}
            # logger.info(f'\n[CHECKPOINT] After Exclude keys matching patterns: {state_dict.keys()}')
        
        # if 'prefix_map' in cfg:
        #     for k,v in cfg.prefix_map.items():
        #         state_dict = {v + kk.removeprefix(k): vv for kk, vv in state_dict.items() if kk.startswith(k)}
        #     logger.info(f'[CHECKPOINT] Applied prefix load map: {cfg.prefix_map}')
        if 'prefix_map' in cfg:
            new_state_dict = {}
            for old_prefix, new_prefix in cfg.prefix_map.items():
                # Find all keys that start with the old prefix
                matching_keys = [k for k in state_dict.keys() if k.startswith(old_prefix)]
                for key in matching_keys:
                    # Replace only the prefix part of the key
                    new_key = new_prefix + key[len(old_prefix):]
                    new_state_dict[new_key] = state_dict[key]
                    # logger.info(f'Mapped {key} -> {new_key}')
            state_dict = new_state_dict
            logger.info(f'[CHECKPOINT] Applied prefix mapping: {cfg.prefix_map}')
            
        logger.info(f'[CHECKPOINT] From "{ckpt_path}" selected keys are: {list(state_dict.keys())}')
        full_state_dict.update(state_dict)

    msg = model.load_state_dict(full_state_dict, strict=False)

    # pretty print of message
    pstr = ''
    for k in ['missing_keys', 'unexpected_keys']:
        pstr += f'{k} (len={len(msg.__getattribute__(k))}):\n'
        for kk in msg.__getattribute__(k):
            pstr += f'  {kk}\n'
    logger.info(f"Loaded state_dict with msg:\n{pstr}")



def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommitted changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


class CosineScheduler(object):
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters, f'{len(self.schedule)}, {self.total_iters}'

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]
        
class CosineSchedulerWithHold(CosineScheduler):
    """
    Add a hold period after warmup and before cosine decay.
    Typically used when you want to hold the learning rate for a few iterations before starting the cosine decay.
    
    """

    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0, hold_iters=0):
        super().__init__(base_value, final_value, total_iters, warmup_iters, start_warmup_value, freeze_iters)
        self.hold_iters = hold_iters

        freeze_schedule = np.zeros((freeze_iters))
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
        hold_schedule = np.ones((hold_iters)) * base_value

        iters = np.arange(total_iters - warmup_iters - freeze_iters - hold_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, hold_schedule, schedule))
        logger.info('Scheduler phases: freeze: {}, warmup: {}, hold: {}, cosine: {}'.format(freeze_schedule.shape, warmup_schedule.shape, hold_schedule.shape, schedule.shape))
        
        # fastdevrun triggers this because OFFICIAL_EPOCH_LENGTH = 10, removing
        # assert len(self.schedule) == self.total_iters, f'{len(self.schedule)}, {self.total_iters}'


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False

import math

class AlphaScheduler(object):
    def __init__(self, 
                 total_steps: int,
                 warmup_steps: int,
                 schedule_type: str = 'cosine',
                 start_alpha: float = 1.0,
                 min_alpha: float = 0.0,
                 epsilon: float = 1e-6,
                 slowdown_threshold: float = 0.,
                 slowdown_fraction: float = 0.1,):
        """
        Args:
            total_steps: Total number of training steps
            warmup_steps: Number of warmup steps where alpha=1
            schedule_type: Type of schedule ('linear', 'cosine', or 'exponential')
            min_alpha: Minimum value of alpha (default 0.0)
            epsilon: Small value to prevent zero gradient
        """
        super().__init__()
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.schedule_type = schedule_type
        self.min_alpha = min_alpha
        self.start_alpha = start_alpha
        self.epsilon = epsilon
        self.slowdown_threshold = slowdown_threshold
        self.slowdown_fraction = slowdown_fraction if slowdown_threshold > 0. else 0.

    def __getitem__(self, step):
        """Calculate alpha with properly scaled slowdown phase.
        """
        if step < self.warmup_steps:
            return self.start_alpha
            
        # Calculate regular progress
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        
        # Calculate base alpha according to chosen schedule
        if self.schedule_type == 'linear':
            alpha = self.start_alpha * (1.0 - progress)
        elif self.schedule_type == 'cosine':
            alpha = self.start_alpha * 0.5 * (1.0 + math.cos(math.pi * progress))
        elif self.schedule_type == 'exponential':
            alpha = self.start_alpha * math.exp(-5 * progress)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        # Apply threshold-based slowdown
        if alpha <= self.slowdown_threshold:
            # Scale the remaining decrease by the current alpha value
            # This makes the decrease proportional to how far we are from zero
            decay_rate = 5.0
            remaining = (self.slowdown_threshold - self.min_alpha) * math.exp(decay_rate * (alpha - self.slowdown_threshold))
            alpha = max(self.min_alpha, remaining)
        
        return alpha