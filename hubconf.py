# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import os
from dinov2.models import vision_transformer as vits


def panopticon_vitb14(dir_to_save_ckpt_in: str = None):

    # download weights
    base_dir = dir_to_save_ckpt_in or torch.hub.get_dir()
    URL = "https://huggingface.co/lewaldm/panopticon/resolve/main/{}"
    f = 'panopticon_vitb14_teacher.pth'
    path_to_ckpt = os.path.join(base_dir, f)
    torch.hub.download_url_to_file(URL.format(f), path_to_ckpt)

    # create model
    teacher = _panopticon_vitb14()

    # load weights
    ckpt = torch.load(path_to_ckpt, map_location="cpu")
    teacher.load_state_dict(ckpt, strict=True)

    return teacher

 
def _panopticon_vitb14():

    arch = 'vit_base'
    vit_kwargs = dict(
        img_size = 518, # was trained on 224 but 568 is legacy from last large image size epoch in dinov2 training
        patch_size = 14,
        init_values = 1.0e-05,
        ffn_layer = 'mlp',
        block_chunks = 0,
        qkv_bias = True,
        proj_bias = True,
        ffn_bias = True,
        num_register_tokens = 0,
        embed_layer = 'PanopticonPE',
        pe_args = dict(
            attn_dim = 2304,
            chnfus_cfg = dict(
                layer_norm = False,
                attn_cfg = dict(
                    num_heads = 16,
                )
            )
        ),
    )

    teacher = vits.__dict__[arch](**vit_kwargs)
    return teacher