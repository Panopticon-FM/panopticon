from omegaconf import OmegaConf
import os
import pandas as pd
import logging
import itertools
import torch.distributed as dist
import time
from dinov2.eval.eval import collect_csv, _distribute_and_execute, build_task
from dinov2.eval.setup import parse_config_obj
import dinov2.distributed as distributed

logger = logging.getLogger


def do_offline_eval(
        base_dir, 
        config_obj, 
        remove_ckpts = False,
        overwrite = False,
        overwrites = None, # overwrites for eval
        summary_logger = None
    ):
    overwrites = overwrites or {}

    # get model config

    train_config = OmegaConf.load(os.path.join(base_dir, 'config.yaml'))
    model_kwargs = train_config.student
    eval_base_dir = os.path.join(base_dir, 'eval')

    # get model_tuples
    
    model_tuples = []
    if os.path.exists(eval_base_dir):
        for d in os.listdir(eval_base_dir):
            if os.path.exists(os.path.join(eval_base_dir, d, 'teacher_checkpoint.pth')):

                output_dir = os.path.join(eval_base_dir, d)
                ckpt_path = os.path.join(eval_base_dir, d, 'teacher_checkpoint.pth')
                
                model_cfg = OmegaConf.create(dict(
                    id = 'panopticon',
                    pretrained_weights = [dict(
                        path=ckpt_path,
                        checkpoint_key='teacher',
                        exclude='cached',
                        prefix_map = {'backbone.': ''})],
                    model_kwargs = model_kwargs))
                
                model_tuples.append((output_dir, model_cfg, output_dir, ''))

    # get configs

    task_tuples = parse_config_obj(config_obj)
    task_tuples = [(c[0], build_task(c[1], **overwrites)) for c in task_tuples]

    # execute

    _distribute_and_execute(
        model_tuples, 
        task_tuples, 
        overwrite=overwrite, 
        distribute='runs', 
        config_obj_str=config_obj, 
        barrier=True,
        summary_logger=summary_logger)

    # gather results

    all_results = []
    if distributed.is_main_process():

        for output_dir, _, _, _ in model_tuples:

            df = collect_csv(output_dir, return_df='relpath')
            iteration = int(os.path.basename(os.path.normpath(output_dir)))

            metric_dict = {}
            for i in range(len(df)):
                relpath = df.at[i,'relpath']
                k = os.path.join(relpath, df.at[i,'metric'])
                v = df.at[i,'value']
                metric_dict[k] = v

            all_results.append((iteration, metric_dict))

            if remove_ckpts:
                os.remove(os.path.join(output_dir, 'teacher_checkpoint.pth'))

    return all_results