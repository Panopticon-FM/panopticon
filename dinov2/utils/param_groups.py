# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from collections import defaultdict
import logging
import re

logger = logging.getLogger("dinov2")


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12, force_is_backbone=False, chunked_blocks=False):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone") or force_is_backbone:
        if (
            ".pos_embed" in name
            or ".patch_embed" in name
            or ".mask_token" in name
            or ".cls_token" in name
            or ".register_tokens" in name
        ):
            layer_id = 0
        elif force_is_backbone and (
            "pos_embed" in name
            or "patch_embed" in name
            or "mask_token" in name
            or "cls_token" in name
            or "register_tokens" in name
        ):
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
        elif chunked_blocks and "blocks." in name and "residual." not in name:
            layer_id = int(name[name.find("blocks.") :].split(".")[2]) + 1
        elif "blocks." in name and "residual." not in name:
            layer_id = int(name[name.find("blocks.") :].split(".")[1]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_params_groups_with_decay(model, lr_decay_rate=1.0, freeze_weights='', lr_multiplier='', prefix=''):
    chunked_blocks = False
    if hasattr(model, "n_blocks"):
        logger.info("chunked fsdp")
        n_blocks = model.n_blocks
        chunked_blocks = model.chunked_blocks
    elif hasattr(model, "blocks"):
        logger.info("first code branch")
        n_blocks = len(model.blocks)
    elif hasattr(model, "backbone"):
        logger.info("second code branch")
        n_blocks = len(model.backbone.blocks)
    else:
        logger.info("else code branch")
        n_blocks = 0
    all_param_groups = []

    def proc_str(string):
        items = [s for s in string.split(',') if s.strip() != '']
        return [s.split('=') for s in items]

    freeze_weights = proc_str(freeze_weights)
    lr_multiplier = proc_str(lr_multiplier)

    for name, param in model.named_parameters():
        name = name.replace("_fsdp_wrapped_module.", "")
        name = prefix + '.' + name
        if not param.requires_grad:
            continue
        decay_rate = get_vit_lr_decay_rate(
            name, lr_decay_rate, num_layers=n_blocks, force_is_backbone=n_blocks > 0, chunked_blocks=chunked_blocks
        )
        d = {"params": param, 'freeze_epochs': 0, "lr_multiplier": decay_rate, "wd_multiplier": 1.0, "name": name}

        for fw_cfg in freeze_weights:
            if re.search(fw_cfg[0], name):
                d.update({"freeze_epochs": int(fw_cfg[1])})

        if name.endswith(".bias") or "norm" in name or "gamma" in name:
            d.update({"wd_multiplier": 0.0})

        for lr_cfg in lr_multiplier:
            if re.search(lr_cfg[0], name):
                d.update({"lr_multiplier": d["lr_multiplier"] * float(lr_cfg[1])})

        all_param_groups.append(d)
        logger.info(f"""{name}: lr_multiplier: {d["lr_multiplier"]}, wd_multiplier: {d["wd_multiplier"]}, freeze_epochs: {d["freeze_epochs"]}""")

    return all_param_groups


def fuse_params_groups(all_params_groups, keys=("lr_multiplier", "wd_multiplier", "freeze_epochs")):
    fused_params_groups = defaultdict(lambda: {"params": []})
    for d in all_params_groups:
        identifier = ""
        for k in keys:
            identifier += k + str(d[k]) + "_"

        for k in keys:
            fused_params_groups[identifier][k] = d[k]
        fused_params_groups[identifier]["params"].append(d["params"])

    return fused_params_groups.values()
