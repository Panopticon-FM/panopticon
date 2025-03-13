# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch

from dinov2.utils.config import resolve_configs, write_config
import logging
from dinov2.eval.wrapper import build_model_for_eval
import sys
import os
from omegaconf import OmegaConf, DictConfig
import glob
import wandb

logger = logging.getLogger("dinov2")


def merge_cfgs(*cfgs):
    merged_cfgs = OmegaConf.merge(*cfgs)
    # print('merged_cfgs', OmegaConf.to_yaml(merged_cfgs))
    resolved_cfg = resolve_configs([merged_cfgs])[0]
    # print('resolved_cfg', OmegaConf.to_yaml(resolved_cfg))
    return resolved_cfg


def _setup(model_cfg, cfg, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    setup_logger('dinov2', os.path.join(output_dir, 'log'), reset_logger=True)
    cfg.output_dir = output_dir
    write_config(cfg, output_dir)
    logger.info(OmegaConf.to_yaml(cfg))

    model = build_model_for_eval(model_cfg)
    return model


# @functools.lru_cache()
def setup_logger(name, filename=None, level=logging.DEBUG, to_sysout=False, simple_prefix=False, reset_logger=True, hard_close=False):

    logger = logging.getLogger(name)
    if reset_logger:
        if hard_close:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()
        else:
            logger.handlers = [] # removeHandler will close handler but we later need it again!
    logger.setLevel(level)
    logger.propagate = False

    if simple_prefix:
        fmt_prefix = "%(asctime)s %(filename)s:%(lineno)s] "
        datefmt = "%H:%M:%S"
    else:
        fmt_prefix = "%(levelname).1s%(asctime)s %(process)s %(name)s %(filename)s:%(lineno)s] "
        datefmt = "%Y%m%d %H:%M:%S"
        
    fmt_message = "%(message)s"
    fmt = fmt_prefix + fmt_message
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    if filename:
        handler = logging.StreamHandler(open(filename, "a+"))
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if to_sysout:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def create_wandb_dir_for_baseline(output_dir):
    """ create wandb run for baseline in its directory """
    name = os.path.relpath(output_dir, os.environ['ODIR'])
    config =  OmegaConf.to_container(OmegaConf.load(os.path.join(output_dir, 'config.yaml')))
    config.train.short_exp_name = name
    wandb.init(
        project=os.environ['WANDB_PROJECT'], 
        name=name, 
        dir = output_dir, 
        config = config)
    wandb.finish()


def parse_model_obj(model_obj):
    """ output is list of model_tuples with: 
            - output_dir [str]: string to reproduce the model_tuple with parse_model_obj(model_obj)
            - model_cfg [OmegaConf]: model config to build model with build_model_for_eval
            - model_str [str]: string to reproduce the model_tuple with parse_model_obj(model_str)
            - wandb_dir [str]: directory to save wandb logs
    """

    # if isinstance(model_obj, DictConfig):
    #     return [(False, f'(custom config of {model_obj.id})', model_obj)]

    if isinstance(model_obj, str):

        def multiple(model_obj):
            model_tuples_list = [parse_model_obj(m) for m in model_obj]
            model_tuples = [m for sublist in model_tuples_list for m in sublist]
            return model_tuples

        if ';' in model_obj:
            model_obj = model_obj.split(';')
            if len(model_obj[-1]) == 0:
                model_obj.pop(-1)
            return multiple(model_obj)

        elif os.path.exists(model_obj):
            
            if os.path.isfile(model_obj) or 'model_final.pth' in os.listdir(model_obj): # is a run_dir from dinov2/train/train.py or ckpt

                if os.path.isfile(model_obj):
                    assert model_obj.endswith('.pth'), 'Not a valid model ckpt'
                    path_to_model_dir = os.path.dirname(model_obj)
                    ckpt_path = model_obj
                else:
                    path_to_model_dir = model_obj
                    ckpt_path = os.path.join(path_to_model_dir, 'model_final.pth')
                output_dir_name = f"eval_{os.path.basename(ckpt_path).removesuffix('.pth')}"

                pretrained_weights = dict(
                    path = ckpt_path,
                    checkpoint_key = 'model',
                    include = 'teacher',
                    exclude = 'cached_optical_embs',
                    prefix_map = {'teacher.backbone.': ''},)

                model_kwargs = OmegaConf.load(os.path.join(path_to_model_dir, 'config.yaml')).student
                if 'rgb_pe_distil' in model_kwargs.pe_args:
                    model_kwargs.pe_args.pop('rgb_pe_distil')
                    logger.info(f'Removed rgb_pe_distil from model {path_to_model_dir}')
                model_cfg = OmegaConf.create(dict(
                    id = 'panopticon',
                    pretrained_weights = [pretrained_weights],
                    model_kwargs = model_kwargs))

                output_dir = os.path.join(path_to_model_dir, output_dir_name)
                model_str = path_to_model_dir 
                wandb_dir = path_to_model_dir
                return [(output_dir, model_cfg, model_str, wandb_dir)]

            elif 'evalme' in os.listdir(model_obj): # other model directory

                path_to_model_dir = model_obj
                model_cfg = OmegaConf.load(os.path.join(path_to_model_dir, 'config.yaml'))
                model_cfg._abspath = os.path.normpath(os.path.abspath(path_to_model_dir))
                OmegaConf.resolve(model_cfg)

                if not os.path.exists(os.path.join(path_to_model_dir, 'wandb')):
                    create_wandb_dir_for_baseline(path_to_model_dir)

                output_dir = os.path.join(path_to_model_dir, 'eval_model_final')
                model_str = path_to_model_dir 
                wandb_dir = path_to_model_dir
                return [(output_dir, model_cfg, model_str, wandb_dir)]

            else: # recursive

                subdirs = [os.path.join(model_obj,d) for d in os.listdir(model_obj) \
                        if os.path.isdir(os.path.join(model_obj,d))]

                if len(subdirs) == 0:
                    return []

                model_tuples = multiple(subdirs)
                
                return model_tuples


        else: # match with glob
            model_tuples = multiple(glob.glob(model_obj))
            if len(model_tuples) == 0:
                raise ValueError(f'No run dirs found in or this is not a valid path: {model_obj}')
            return model_tuples
    else:
        raise ValueError(f'Unknown model_obj type {type(model_obj)}')


def parse_config_obj(config_obj):
    """ parse a config obj into list of tuples with:
            - relpath [str]: relative path to the config file serving as id for the task
            - cfg [OmegaConf]: config object """

    if isinstance(config_obj, str):
        config_obj = config_obj.split(';')
        if len(config_obj[-1]) == 0:
            config_obj.pop(-1)
        config_obj = [glob.glob(c) for c in config_obj]
        config_obj = [os.path.normpath(c) for sublist in config_obj for c in sublist]
        config_root_dir = os.path.join(os.environ['CDIR'],'eval')

        tasks = []
        for root_path in config_obj:

            # single file
            if os.path.isfile(root_path):
                if root_path.endswith('.yaml'):
                    cfg = OmegaConf.load(root_path)
                    relpath = os.path.relpath(root_path, config_root_dir)[:-5]
                    tasks.append((relpath, cfg))
                else:
                    raise ValueError(f'Unknown file type {root_path}')

            # directory
            for root, dir, files in os.walk(root_path):
                yaml_file = [f for f in files if f.endswith('.yaml')]

                for f in yaml_file:
                    cfg_path = os.path.join(root, f)
                    relpath = os.path.relpath(cfg_path, config_root_dir)[:-5]
                    cfg = OmegaConf.load(cfg_path)
                    tasks.append((relpath, cfg))
        return tasks
    
    elif config_obj is None:
        return []
    
    else: 
        raise ValueError(f'Unknown config_obj {config_obj}')

