# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import copy

from . import vision_transformer as vits

logger = logging.getLogger("dinov2")


def build_model(args, 
                only_teacher=False, 
                img_size=224, 
                student_pretrained_weights=[],
                teacher_pretrained_weights=[]):
    args.arch = args.arch.removesuffix("_memeff")
    from omegaconf import OmegaConf
    from dinov2.utils.utils import load_pretrained_weights

    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
            embed_layer=args.embed_layer,
            pe_args = OmegaConf.to_container(OmegaConf.create(args.get('pe_args', {})), resolve=True),
            in_chans=args.in_chans,
        )
        #create a copy of kwargs to guard against changes in the original dict
        student_vit_kwargs = copy.deepcopy(vit_kwargs)

        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if len(teacher_pretrained_weights) > 0: # not super clean since weights might be overwritten by loading weights for full ssl_meta_arch, but okay for now
            logger.info(f"Loading teacher weights ...")
            load_pretrained_weights(teacher, teacher_pretrained_weights)
        if only_teacher:
            return teacher, teacher.embed_dim

        student = vits.__dict__[args.arch](
            **student_vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        if len(student_pretrained_weights) > 0:
            logger.info(f"Loading student weights ...")
            load_pretrained_weights(student, student_pretrained_weights)
        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    student_pretrained_weights = cfg.student.pretrained_weights
    if only_teacher or cfg.teacher.pretrained_weights=='same_as_student':
        logger.info("Using student weights for teacher")
        teacher_pretrained_weights = student_pretrained_weights
    else:
        teacher_pretrained_weights = cfg.teacher.pretrained_weights

    return build_model(cfg.student,
                        only_teacher=only_teacher,
                        teacher_pretrained_weights=teacher_pretrained_weights,
                        student_pretrained_weights=student_pretrained_weights,
                        img_size=cfg.student.pos_emb_img_size,)
