from omegaconf import OmegaConf
import torch.nn as nn
import torch
from torchvision.transforms import Resize

from dinov2.models import build_model_from_cfg
from dinov2.utils.config import resolve_configs
from dinov2.configs import dinov2_default_config
import os
from torchvision.datasets.utils import download_url
import logging
from dinov2.utils.utils import load_pretrained_weights

logger = logging.getLogger("dinov2")


def build_model_for_eval(config):
    """ build actual pytorch model from model_cfg"""

    id = config.id
    pretrained_weights = config.pretrained_weights
    model_kwargs = config.model_kwargs
    autocast_dtype_str = config.get('autocast_dtype_str', 'fp16')
    resize = config.get('resize', None)

    # build model

    model_registry = {
        'panopticon': PanopticonWrapper,

    }

    model = model_registry[id](model_kwargs=model_kwargs)
    logger.info(f"Built model {id}")

    autocast_dtype = get_autocast_dtype(autocast_dtype_str)
    model._prepare(pretrained_weights=pretrained_weights, autocast_dtype=autocast_dtype, resize=resize)

    return model

def get_autocast_dtype(dtype_str):
    if dtype_str == "fp16":
        return torch.half
    elif dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float32

def backbone_to_features(blocks_list, use_n_blocks, pooling, norm:nn.Module=None):
    blocks_list = blocks_list[-use_n_blocks:] # list of output tensors of last n blocks

    if pooling == 'avgpool': # corresponds to DINOv2 avgpool=True
        blocks_list = [norm(x) for x in blocks_list]
        class_tokens = [x[:, 0] for x in blocks_list]

        output = torch.cat(
            (
                torch.cat(class_tokens, dim=-1),
                torch.mean(blocks_list[-1][:, 1:], dim=1),
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)

    elif pooling == 'cls': # corresponds to DINOv2 avgpool=False
        blocks_list = [norm(x) for x in blocks_list]
        class_tokens = [x[:, 0] for x in blocks_list]
        output = torch.cat(class_tokens, dim=-1)

    elif pooling == 'DOFA_globalpool':
        output = blocks_list[-1][:, 1:].mean(dim=1)
        output = norm(output) 

    elif pooling == 'DOFA_no_globalpool':
        output = blocks_list[-1][:, 0]
        output = norm(output.unsqueeze(0)).squeeze(0)

    elif pooling == 'knn': # consistent with vanilla DINOv2 eval knn
        output = norm(blocks_list[-1])[:,0]
        output = nn.functional.normalize(output, dim=1, p=2) # big performance drop if not used!

    elif pooling == 'lin': # consistent with vanilla DINOv2 .forward()
        output = norm(blocks_list[-1])[:,0]

    else:
        raise ValueError(f"Pooling {pooling} not supported")

    return output.float()


class ModelWithIntermediateLayers(nn.Module):
    """ base eval class"""

    feature_model: nn.Module
    n_last_blocks: int
    # bb_to_feat_adapter = None
    # norm = None # norm is handeled in create_linear_input in LinearClassifier

    def _prepare(self,  
                 pretrained_weights = None, 
                 autocast_dtype=None, 
                 resize=None):
        """ needs to be called before the first forward pass"""

        self.autocast_dtype = autocast_dtype
        self.return_class_token = True

        assert isinstance(self.feature_model, nn.Module), 'feature_model must be a nn.Module'
        if len(pretrained_weights) > 0:
            load_pretrained_weights(self.feature_model, pretrained_weights)
        else:
            logger.warning('No pretrained weights specified. Model will be randomly initialized.')

        for p in self.feature_model.parameters():
            p.requires_grad = False
        if resize is not None:
            self.resize = Resize(size=resize, antialias=True)
    
        self.feature_model.eval()
        self.feature_model.cuda()

    def set_n_last_blocks(self, n_last_blocks):
        self.n_last_blocks = n_last_blocks

    def get_intermediate_layers(self, x_dict):
        raise NotImplementedError()

    def forward(self, x_dict):

        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=self.autocast_dtype is not None, dtype=self.autocast_dtype):
                if hasattr(self, 'resize'):
                    x_dict['imgs'] = self.resize(x_dict['imgs'])
                
                features = self.get_intermediate_layers(x_dict)
                if hasattr(self, 'bb_to_feat_adapter'):
                    features = self.bb_to_feat_adapter(features, norm=self.norm)
        return features


class PanopticonWrapper(ModelWithIntermediateLayers):

    def __init__(self, model_kwargs):
        super().__init__()

        cfgs = [
            OmegaConf.create(dinov2_default_config).student,
            model_kwargs] 
        build_cfg = {'student': OmegaConf.merge(*resolve_configs(cfgs))}
        build_cfg = OmegaConf.create(build_cfg)
        model, _ = build_model_from_cfg(build_cfg, only_teacher=True)
        self.feature_model = model

        self.norm = self.feature_model.norm

    def _prepare(self, *args, **kwargs):
        autocast_dtype = kwargs['autocast_dtype']
        assert autocast_dtype == torch.half, 'autocast_dtype_str set to fp16 for DINOv2 model.'
        super()._prepare(*args, **kwargs)

    def get_intermediate_layers(self, x_dict):
        # just copied from dinov2/models/vision_transformer.DinoVisionTransformer.get_intermediate_layers
        if self.feature_model.chunked_blocks:
            outputs = self.feature_model._get_intermediate_layers_chunked(x_dict, self.n_last_blocks)
        else:
            outputs = self.feature_model._get_intermediate_layers_not_chunked(x_dict, self.n_last_blocks)
        return outputs
        # return self.feature_model.get_intermediate_layers(
        #     x_dict, self.n_last_blocks, return_class_token=self.return_class_token, reshape=self.reshape)