# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
from functools import partial

from fvcore.common.checkpoint import PeriodicCheckpointer
from fvcore.common.checkpoint import Checkpointer
import torch

from dinov2.eval.setup import setup_logger
from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data import collate_data_and_cast, MaskingGenerator
import dinov2.distributed as distributed
from dinov2.logging import MetricLogger
from dinov2.utils.config import train_setup
from dinov2.utils.utils import CosineScheduler, load_pretrained_weights, AlphaScheduler, CosineSchedulerWithHold

from dinov2.train.ssl_meta_arch import SSLMetaArch
import time
import traceback
import shutil
from dinov2.eval.offline_train_eval import do_offline_eval
from dinov2.eval.eval import collect_csv, main as eval_main
import wandb
import datetime
import torch.distributed as dist

from dotenv import load_dotenv
load_dotenv()


torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--env", default=None, help='overwrite config with env specific vars')
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. ",)
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )
    parser.add_argument(
        '--fastdevrun',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Wheter to do a quick debug run with tmp directory, no wandb, small batches'
    )

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
        freeze_iters=cfg.optim.get("freeze_epochs", 0) * OFFICIAL_EPOCH_LENGTH,
        hold_iters=cfg.optim.get("hold_epochs", 0) * OFFICIAL_EPOCH_LENGTH,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineSchedulerWithHold(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, epoch):
    for param_group in optimizer.param_groups:
        freeze_epochs = param_group["freeze_epochs"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (0 if epoch < freeze_epochs else lr) * lr_multiplier


def save_teacher(cfg, model, iteration):
    logger.info(f'Saving teacher for post-training eval ... ')
    eval_time = time.time()
    new_state_dict = model.teacher.state_dict()

    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)

        # save teacher checkpoint
        teacher_ckpt_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckpt_path)
        logger.info(f'Saved teacher weights to {teacher_ckpt_path}')
    
    eval_time = time.time() - eval_time 
    logger.info(f'Saving teacher done ({eval_time:.2f}s)')

def format_td(td: datetime.timedelta):
    s = ''
    if td.days > 0:
        s += f"{td.days}d "
    if td.seconds//3600 > 0:
        s += f"{td.seconds//3600}h "
    s += f"{(td.seconds//60)%60}m"
    return s

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def do_train(cfg, model, resume=True):
    assert not cfg.eval.use_online_eval, 'Online evaluation not supported yet. Please use offline evaluation.'
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training
    use_online_eval = cfg.eval.use_online_eval
    use_wandb = cfg.train.use_wandb and distributed.is_main_process()

    # setup datasets

    pretrain_augm_cfg = cfg.train.pretrain_augm
    dataset = make_dataset(cfg.train.dataset, pretrain_augm_cfg=pretrain_augm_cfg, seed=cfg.train.seed)
    model.global_crops_number = pretrain_augm_cfg.global_crops_number
    model.local_crops_number = pretrain_augm_cfg.local_crops_number

    if cfg.train.OFFICIAL_EPOCH_LENGTH == -1:
        cfg.train.OFFICIAL_EPOCH_LENGTH = math.ceil(
            len(dataset) / (cfg.train.batch_size_per_gpu * distributed.get_global_size()))
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH

    # setup optimizer

    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
    ) = build_schedulers(cfg)

    # setup checkpointer

    checkpointer = Checkpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)
    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    if not checkpointer.has_checkpoint():
        start_iter = 0
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH
    break_at_iter = max_iter
    if cfg.optim.break_at_epochs > 0:
        break_at_iter = cfg.optim.break_at_epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=math.ceil(cfg.train.saveckp_freq * OFFICIAL_EPOCH_LENGTH),
        max_iter=break_at_iter,
        max_to_keep=20,
    )

    # setup data loader

    if isinstance(pretrain_augm_cfg.global_crops_size, int):
        img_size = pretrain_augm_cfg.global_crops_size
    else:
        img_size = tuple(pretrain_augm_cfg.global_crops_size)[1]
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    sampler_type = SamplerType.INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        drop_last=cfg.train.drop_last,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers,
        shuffle=True,
        seed=start_iter,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        collate_fn=collate_fn,
    )

    #check if cfg.optim contains repa
    if 'repa' in cfg.optim:
        logger.info('Using repa')
        #setup alpha scheduler for rep alignment in PE
        alpha_scheduler = AlphaScheduler(
            total_steps=cfg.optim.repa["alpha_total_epochs"] * OFFICIAL_EPOCH_LENGTH,
            warmup_steps=cfg.optim.repa["alpha_warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
            schedule_type=cfg.optim.repa["alpha_schedule_type"],
            start_alpha=cfg.optim.repa["alpha_start"],
            min_alpha=cfg.optim.get("alpha_end", 0.0),
            epsilon=1e-6
        )

    # setup training loop

    if hasattr(dataset, 'is_webdataset') and dataset.is_webdataset:
        dataset_len = dataset.num_samples
    else:
        dataset_len = len(dataset)

    iteration = start_iter
    epoch = iteration // OFFICIAL_EPOCH_LENGTH
    if model.do_grad_accumulation:
        eff_bsz = cfg.train.batch_size_per_gpu * distributed.get_global_size() * model.gradient_accumulation_steps
    else:
        eff_bsz = cfg.train.batch_size_per_gpu * distributed.get_global_size()
    
    logger.info(
        f'#########################################################\n'
        f'#nsamples={dataset_len}, '
        f'batch_size_per_gpu={cfg.train.batch_size_per_gpu}, '
        f'num_gpus={distributed.get_global_size()}, '
        f'grad_accum_steps={cfg.get("gradient_accumulation_steps", 1)}, '
        f'eff_bsz={eff_bsz}, '
        f'OFFICIAL_EPOCH_LENGTH={OFFICIAL_EPOCH_LENGTH}'
        f'#########################################################\n'
    )
    logger.info(f"Starting training from: iteration={iteration}, epoch={epoch}\n")
    metric_logger = MetricLogger(delimiter="  ", 
                                 output_dir=cfg.train.output_dir, 
                                 output_file = 'training_metrics.json',
                                 use_wandb=use_wandb)
    header = "Training"

    print_trainable_parameters(model)

    period_epoch = cfg.eval.eval_period_epoch
    period_iter = cfg.eval.eval_period_iterations
    if period_epoch > 0 and period_iter > 0:
        raise ValueError("Only one of period_epoch or period_iter can be set.")
    elif period_iter < 0 and period_epoch < 0:
        raise ValueError("Either period_epoch or period_iter must be set.")
    elif period_epoch > 0:
        eval_period_iterations = period_epoch * OFFICIAL_EPOCH_LENGTH
    else: 
        eval_period_iterations = period_iter

    # training loop

    train_time = time.time()
    for data in metric_logger.log_every(
        data_loader,
        cfg.train.log_every_n_steps,
        header,
        max_iter,
        iteration,
        use_self_timemeters=True,
        epoch_len = OFFICIAL_EPOCH_LENGTH,
        nsamples_per_iter=eff_bsz,
        dataset_len=dataset_len,
    ):
        if cfg.eval.only_eval: # quick fix for now, cannot not enter loop since need metric_logger setup
            break
        current_batch_size = data["collated_global_crops"]['imgs'].shape[0] / 2
        if iteration >= max_iter:
            logger.info('Max iterations reached. Exiting training loop.')
            break
        if iteration >= break_at_iter:
            logger.info('Ending training early because of break_at_epochs arg. Exiting training loop.')
            break

        # apply schedules

        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        if cfg.optim.online_lr_batch_scaling:
            lr_scale = \
                math.sqrt(current_batch_size * distributed.get_global_size() / 1024) / \
                math.sqrt(cfg.train.batch_size_per_gpu * distributed.get_global_size() / 1024.0)
            lr *= lr_scale
        apply_optim_scheduler(optimizer, lr, wd, epoch)
        
        #rep align alpha
        if 'repa' in cfg.optim:
            current_alpha = alpha_scheduler[iteration]
            if hasattr(model.student.backbone.patch_embed.chnfus, 'vanilla_pe'):
                model.student.backbone.patch_embed.chnfus.set_alpha(current_alpha)

        # compute losses
        # Only zero gradients at start of new accumulation cycle
        if not model.do_grad_accumulation or model.current_accumulation_step == 0:
            optimizer.zero_grad(set_to_none=True)
        # optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

        # clip gradients

        # if fp16_scaler is not None:
        #     if cfg.optim.clip_grad:
        #         fp16_scaler.unscale_(optimizer)
        #         for v in model.student.values():
        #             v.clip_grad_norm_(cfg.optim.clip_grad)
        #     fp16_scaler.step(optimizer)
        #     fp16_scaler.update()
        # else:
        #     if cfg.optim.clip_grad:
        #         for v in model.student.values():
        #             v.clip_grad_norm_(cfg.optim.clip_grad)
        #     optimizer.step()
        if fp16_scaler is not None:
            if model.do_grad_accumulation:
                # Only step if we've accumulated enough gradients
                model.current_accumulation_step += 1
                if model.current_accumulation_step >= model.gradient_accumulation_steps:
                    # Only unscale and clip when we're about to step
                    if cfg.optim.clip_grad:
                        fp16_scaler.unscale_(optimizer)
                        for v in model.student.values():
                            v.clip_grad_norm_(cfg.optim.clip_grad)
                    fp16_scaler.step(optimizer)
                    fp16_scaler.update()
                    model.current_accumulation_step = 0
            else:
                # Original non-accumulation behavior
                if cfg.optim.clip_grad:
                    fp16_scaler.unscale_(optimizer)
                    for v in model.student.values():
                        v.clip_grad_norm_(cfg.optim.clip_grad)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
        else:
            if model.do_grad_accumulation:
                # Only step if we've accumulated enough gradients
                model.current_accumulation_step += 1
                if model.current_accumulation_step >= model.gradient_accumulation_steps:
                    if cfg.optim.clip_grad:
                        for v in model.student.values():
                            v.clip_grad_norm_(cfg.optim.clip_grad)
                    optimizer.step()
                    model.current_accumulation_step = 0
            else:
                if cfg.optim.clip_grad:
                    for v in model.student.values():
                        v.clip_grad_norm_(cfg.optim.clip_grad)
                optimizer.step()


        # perform teacher EMA update
        model.update_teacher(mom)

        # logging

        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            logger.info(loss_dict_reduced)
            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(teacher_temp=teacher_temp)
        metric_logger.update(current_batch_size=current_batch_size)
        
        if model.do_grad_accumulation:
            metric_logger.update(effective_batch_size=eff_bsz)
            metric_logger.update(current_accumulation_step=model.current_accumulation_step)

        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        if model.do_aux_loss:
            # gw = loss_dict.pop('gating_weights')
            # B, L, E = gw.shape
            #very hacky for now
            mean_activations_per_expert = model.student.backbone.patch_embed.get_mean_activations() 
            metric_logger.update_expert_metrics('expert_activations', mean_activations_per_expert.cpu().numpy())


        #repa_alpha
        if 'repa' in cfg.optim:
            metric_logger.update(repa_alpha=current_alpha)

        # checkpointing & testing

        if distributed.is_main_process(): # this only works with DDP not FSDP
            periodic_checkpointer.step(iteration)
        if (eval_period_iterations > 0 and (iteration + 1) % eval_period_iterations == 0) or \
            (iteration + 1 == max_iter and cfg.eval.include_final_ckpt):

            if use_online_eval:
                raise NotImplementedError('Online evaluation not supported yet. Please use offline evaluation.')
            else:
                save_teacher(cfg, model, f"{iteration}")
            torch.cuda.synchronize()
        
        iteration = iteration + 1

    train_time = time.time() - train_time
    if use_wandb and distributed.is_main_process():
        s = wandb.run.summary.get('_t_train', 0) + train_time
        wandb.run.summary['_t_train'] = s
        wandb.run.summary['t_train'] = format_td(datetime.timedelta(seconds=s))

    # do evaluation (design: main_proc waits until all others are finished)

    metric_logger.synchronize_between_processes()
    del model, optimizer, data_loader, dataset, checkpointer, periodic_checkpointer
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(5) # just to be absolutely sure

    # ckpts from train eval

    eval_time = time.time()

    eval_logger = setup_logger(
        'eval', 
        filename = os.path.join(cfg.train.output_dir,'logs','eval.txt'),
        to_sysout = True, 
        reset_logger = True)

    if not cfg.eval.skip and os.path.exists(os.path.join(cfg.train.output_dir,'eval')):
        if distributed.is_main_process():
            eval_logger.info('Offline Evaluation at end of training ... ')
        
        all_results = do_offline_eval(cfg.train.output_dir, 
                                      cfg.eval.config_obj, 
                                      remove_ckpts=cfg.eval.remove_ckpts,
                                      overwrite=cfg.eval.overwrite,
                                      overwrites=cfg.eval.overwrites,
                                      summary_logger = eval_logger)

        if use_wandb and distributed.is_main_process():
            for iteration, metric_dict in all_results:
                metric_logger.log_wandb(iteration, metric_dict, log_step=False, prefix='val')
        dist.barrier() 
        
    elif not cfg.eval.skip and use_online_eval:
        raise NotImplementedError('Online evaluation not supported yet. Please use offline evaluation.')
    
    else:
        if distributed.is_main_process():
            eval_logger.info('No offline evaluation done at end of training')

    # final_model
    if not cfg.eval.skip and cfg.eval.final_model.config_obj is not None:
        if distributed.is_main_process():
            eval_logger.info('Offline Evaluation of final model ... ')

        eval_main(
            model_obj = cfg.train.output_dir, 
            **cfg.eval.final_model, 
            wandb_mode = 'resume',
            barrier = True,
            summary_logger = eval_logger)
        
        if distributed.is_main_process():
            df = collect_csv(os.path.join(cfg.train.output_dir,'eval_model_final'), return_df='lvl')
            eval_logger.info(f'\n{df.to_string()}\n')

    eval_time = time.time() - eval_time
    if use_wandb and distributed.is_main_process():
        s = wandb.run.summary.get('_t_eval', 0) + eval_time
        wandb.run.summary['_t_eval'] = s
        wandb.run.summary['t_eval'] = format_td(datetime.timedelta(seconds=s))

    return os.path.join(cfg.train.output_dir)

def main(args):
    cfg = train_setup(args)
    os.environ['OUTPUT_DIR'] = cfg.train.output_dir

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    if len(cfg.MODEL.pretrained_weights) > 0:
        logger.info('Loading pretrained weights for SSLMetaArch ...')   
        load_pretrained_weights(model, cfg.MODEL.pretrained_weights)
    model.prepare_for_distributed_training()
    logger.info(f'model weight datatype: {model.teacher["backbone"].pos_embed.dtype}')
    logger.info("Model:\n{}".format(model))

    if cfg.tmpfile:
        with open(cfg.tmpfile, 'w') as f:
            f.write(os.path.join(cfg.train.output_dir,'model_final.pth'))

    try: 
        do_train(cfg, model, resume=args.resume)
    except Exception as e:
        logger.error('Error message: ' + str(e))
        logger.error('Full traceback:\n ' + traceback.format_exc())
        raise e
    finally:
        if args.fastdevrun:
            shutil.rmtree(cfg.train.output_dir)
            logger.info('Debug run complete. Removed directory.')
            return

if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    print('overwriting resume=True')
    args.resume = True
    main(args)