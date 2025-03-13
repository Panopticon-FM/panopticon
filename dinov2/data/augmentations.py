# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


import logging

from torchvision import transforms

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)
from random import shuffle
import numpy as np
from torchvision.transforms.functional import pad, resized_crop
import torch
from copy import deepcopy
from torch import Tensor

from typing import List, Tuple, Optional
from omegaconf import OmegaConf

logger = logging.getLogger("dinov2")


def make_augmentation(cfg_list):
    """ builds augmentations and also returns metainfo about crops """
    cfg_list = deepcopy(cfg_list)
    ret, ids = [], []

    for trf_cfg in cfg_list:
        if OmegaConf.is_list(trf_cfg):
            trf = make_augmentation(trf_cfg)
        else:
            id = trf_cfg.pop('id')
            ids.append(id)

            assert id in globals(), f"Augmentation {id} not supported"
            trf = globals()[id](**trf_cfg) # call any class from this file by its name

        ret.append(trf)
    logger.info(f"Augmentations in order: {ids}")

    ret = transforms.Compose(ret)
    return ret


# pretraining augmentations


class CropModes:
    SENSOR_SINGLE_MASKED = 0 # take single view, apply spectral masking
    CHN_SIM = 1 # simulate channels, only for hyperspectral, apply spectral masking

class PanopticonAugmentation:
    """ main pre-training augmentation """

    def __init__(self,

        global_crops_number = 2,
        global_crops_size = 224,
        global_crops_scale = [0.32, 1.0],
        global_crops_spectral_size = [4,13],
        global_multi_select_view_in_single_sensor_modes = False,
        global_hs_modes_probs = [1,0],

        local_crops_number = 4,
        local_crops_scale = [0.05, 0.32],
        local_crops_size = 98,
        local_crops_spectral_size = [1,4],
        local_multi_select_view_in_single_sensor_modes = True,
        local_hs_modes_probs  = [1,0],

        color_jitter_args = dict(p=0.2, brightness=0.1, contrast=0.1, saturation=0.0, hue=0.0),
        hs_valid_wavelength_ranges = [[418, 1282], [1530, 1728], [1977, 2445]],
        hs_valid_sigma_ranges=[(5, 40)],
        hs_sigma_distribution='uniform',
    ):
        
        assert global_crops_number == 2, "Only 2 global crops supported"

        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.global_crops_number = global_crops_number
        self.global_hs_modes_probs = global_hs_modes_probs
        self.local_hs_modes_probs = local_hs_modes_probs
        self.hs_valid_wavelength_ranges = hs_valid_wavelength_ranges
        self.hs_valid_sigma_ranges = hs_valid_sigma_ranges
        self.hs_sigma_distribution = hs_sigma_distribution
        self.global_multi_select_view_in_single_sensor_modes = global_multi_select_view_in_single_sensor_modes
        self.local_multi_select_view_in_single_sensor_modes = local_multi_select_view_in_single_sensor_modes

        logger.info("###################################")
        logger.info(f'id: PanopticonAugmentation')
        logger.info(f'global_crops_number: {global_crops_number}')
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"global_crops_spectral_size: {global_crops_spectral_size}")
        logger.info(f"global_multi_select_view_in_single_sensor_modes: {global_multi_select_view_in_single_sensor_modes}")
        logger.info(f"global_hs_modes_probs: {global_hs_modes_probs}")
        logger.info('')
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_spectral_size: {local_crops_spectral_size}")
        logger.info(f"local_multi_select_view_in_single_sensor_modes: {local_multi_select_view_in_single_sensor_modes}")
        logger.info(f"local_hs_modes_probs: {local_hs_modes_probs}")
        logger.info('')
        logger.info(f"hs_valid_wavelength_ranges: {hs_valid_wavelength_ranges}")
        logger.info(f"hs_valid_sigma_ranges: {hs_valid_sigma_ranges}")
        logger.info(f"hs_sigma_distribution: {hs_sigma_distribution}")
        logger.info("###################################")

        self.crop_modes = CropModes

        # geometric augmentations

        self.global_chn_mask = ListChnMask(*global_crops_spectral_size)
        self.global_geometric_augmentation = transforms.Compose([
            ListRandomResizeCrop(global_crops_size, global_crops_scale),
            RandomHVFlip(p=0.5),
        ])

        self.local_chn_mask = ListChnMask(*local_crops_spectral_size)
        self.local_geometric_augmentation = transforms.Compose([
            ListRandomResizeCrop(local_crops_size, local_crops_scale),
            RandomHVFlip(p=0.5),
        ])

        # chn_sim augmentations (only used in appendix)

        self.global_channel_sim_aug = ChannelSimAugmentation(
            n_channels_range = global_crops_spectral_size[1:],
            valid_wavelength_ranges = hs_valid_wavelength_ranges,
            valid_sigma_ranges = hs_valid_sigma_ranges,
            sigma_distribution = hs_sigma_distribution,)
        
        self.local_channel_sim_aug = ChannelSimAugmentation(
            n_channels_range = local_crops_spectral_size[1:],
            valid_wavelength_ranges = hs_valid_wavelength_ranges,
            valid_sigma_ranges = hs_valid_sigma_ranges,
            sigma_distribution = hs_sigma_distribution)

        # other augmentations

        color_jittering = ColorJitterRS(**color_jitter_args)

        self.global_transfo = transforms.Compose([
            color_jittering,
            ChnPad(global_crops_spectral_size[1]),
            ])
        self.local_transfo = transforms.Compose([
            color_jittering,
            ChnPad(local_crops_spectral_size[1]),
            ])

    def _listoflists2list(self, list_of_lists):
        return [item for sublist in list_of_lists for item in sublist]

    def _append_data_objs(self, data_objs):
        output = {}
        for k in data_objs[0].keys():
            output[k] = self._listoflists2list([d[k] for d in data_objs])
        return output

    def __call__(self, dicts_list: List):
        """ the basic data object (data_obj) is a dict. Each key contains a list of equal length with
                imgs: [L, tensor(c,h,w)]
                chn_ids: [L, tensor(c,...)]
            The input is a list of data_objs, one for each sensor
        """

        def _augment(
                dicts_list, 
                hs_modes_probs,
                crops_number,
                chn_mask,
                channel_sim_aug, 
                geometric_aug, 
                transfo,
                multi_select_view_in_single_sensor_modes,
        ):
            
            # sample kind of modes

            if len(dicts_list) == 1 and dicts_list[0]['imgs'][0].shape[0] > 30: # is single-view hyperspetral
                
                modes = np.random.choice(
                    range(len(hs_modes_probs)), 
                    size = crops_number, 
                    p = hs_modes_probs, 
                    replace = True)
                
                single_sensor_idx = iter(np.random.choice(
                    range(len(dicts_list)), 
                    len(modes), 
                    replace = True)) # replace=True since only single hyperspectral image
                
            else:

                modes = np.zeros(crops_number)

                single_sensor_idx = iter(np.random.choice(
                    range(len(dicts_list)), 
                    len(modes), 
                    replace = multi_select_view_in_single_sensor_modes))

            # augment with selected modes

            crops = []
            
            for mode in modes:
                if mode == self.crop_modes.SENSOR_SINGLE_MASKED:
                    idx = next(single_sensor_idx)
                    data_obj = dicts_list[idx]
                    data_obj = chn_mask(data_obj)

                elif mode == self.crop_modes.CHN_SIM:
                    data_obj = self._append_data_objs(dicts_list)
                    data_obj = channel_sim_aug(data_obj)

                else:
                    raise ValueError(f"Unknown mode: {mode}")

                data_obj = geometric_aug(data_obj)
                data_obj = transfo(data_obj)
                crops.append(data_obj)

            return crops


        global_crops = _augment(
            dicts_list, 
            self.global_hs_modes_probs,
            self.global_crops_number,
            self.global_chn_mask, 
            self.global_channel_sim_aug, 
            self.global_geometric_augmentation,
            self.global_transfo, 
            self.global_multi_select_view_in_single_sensor_modes,)

        local_crops = _augment(
            dicts_list, 
            self.local_hs_modes_probs,
            self.local_crops_number,
            self.local_chn_mask,
            self.local_channel_sim_aug,
            self.local_geometric_augmentation,
            self.local_transfo, 
            self.local_multi_select_view_in_single_sensor_modes,)

        # return

        output = {}
        output["global_crops"] = global_crops
        output["global_crops_teacher"] = global_crops
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output

class ChnSelector:
    """ abstract class to implement transforms that subselect channels of a data object """

    def _get_indices(self, data_obj):
        chn_indices = [(s,i) for s in range(len(data_obj['imgs'])) 
                    for i in range(len(data_obj['imgs'][s]))]
        return chn_indices

    def _apply_selection(self, data_obj, chn_indices):
        keys = data_obj.keys()
        data_obj_out = {k: [] for k in keys}
        for chn_group_in in range(len(data_obj['imgs'])):
            idx = [i for s,i in chn_indices if s == chn_group_in]
            if len(idx) > 0:
                for k in keys:
                    data_obj_out[k].append(data_obj[k][chn_group_in][idx])
        return data_obj_out

class ListChnMask(ChnSelector):
    """ randomly mask out channels """

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, data_obj):
        chn_indices = self._get_indices(data_obj)

        high = min(self.high, len(chn_indices))
        low = self.low

        if len(chn_indices) <= low:
            return data_obj

        nchns = np.random.randint(low, high+1)
        if nchns == len(chn_indices):
            return data_obj

        shuffle(chn_indices)
        chn_indices = chn_indices[:nchns]
        return self._apply_selection(data_obj, chn_indices)

class ColorJitterRS:
    def __init__(self, p=0.3, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1):
        trf = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        trf = transforms.RandomApply([trf], p=p)
        self.trf = trf

    def __call__(self, data_obj):
        data_obj_out = {k: v for k,v in data_obj.items() if k != 'imgs'}
        data_obj_out['imgs'] = self.trf(data_obj['imgs'].unsqueeze(1)).squeeze(1)
        return data_obj_out

class ListRandomResizeCrop:
    """ takes list of different shape images and outputs RandomResizedCrop of same relative location """
    def __init__(self, 
                 size, 
                 scale, 
                 ratio = (3.0 / 4.0, 4.0 / 3.0),
                 antialias = True,
                 interpolation=transforms.InterpolationMode.BICUBIC,
                 ):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.antialias = antialias
        self.interpolation = interpolation

    def __call__(self, x_dict: dict):
        img_list = x_dict['imgs']
        add_kwargs = {k:v for k,v in x_dict.items() if k not in ['imgs', 'chn_ids']}

        # get crop for largest in list
        idx_largest = np.argmax([img.shape[1] for img in img_list])
        i, j, h, w = transforms.RandomResizedCrop.get_params(img_list[idx_largest], self.scale, self.ratio)

        # crop all images (not in_place)
        img_largest_shape = img_list[idx_largest].shape
        ret = []
        for idx in range(len(img_list)):
            img = img_list[idx]
            if idx != idx_largest:
                
                # Calculate the crop parameters for the current image
                scale_x = img.shape[-1] / img_largest_shape[-1]
                scale_y = img.shape[-2] / img_largest_shape[-2]
                i_scaled, j_scaled = int(np.round(i * scale_y)), int(np.round(j * scale_x))
                h_scaled, w_scaled = int(np.ceil(h * scale_y)), int(np.ceil(w * scale_x))

                img_out = resized_crop(img, i_scaled, j_scaled, h_scaled, w_scaled,
                                    [self.size, self.size], self.interpolation, antialias=self.antialias)
      
            else:
                img_out = resized_crop(img, i, j, h, w, 
                                    [self.size, self.size], self.interpolation, antialias=self.antialias)
            ret.append(img_out)

        ret = torch.cat(ret, dim=0)
        chn_ids = x_dict['chn_ids']
        if isinstance(chn_ids, list):
            chn_ids = torch.cat(chn_ids, dim=0)
        return dict(imgs=ret, chn_ids=chn_ids, **add_kwargs)
    
class ChnPad:
    """ pads channels to same size """
    def __init__(self, size):
        self.size = size

    def __call__(self, x_dict):
        if self.size < 0:
            return x_dict
        img: Tensor = x_dict['imgs']
        chn_ids = x_dict['chn_ids']
        add_kwargs = {k:v for k,v in x_dict.items() if k not in ['imgs', 'chn_ids']}
        device = img.device

        c, h, w = img.shape
        spec_masks = torch.zeros(c, device=device)
        if c < self.size:
            img = torch.cat([img, torch.zeros(self.size - c, h, w, device=device)])
            # Check the number of dimensions
            if chn_ids.ndimension() == 1:
                chn_ids = torch.cat([chn_ids, torch.full((self.size - c,), chn_ids[-1])])
            elif chn_ids.ndimension() == 2:
                # Repeat the last row to fill the remaining size
                last_row = chn_ids[-1].unsqueeze(0).repeat(self.size - c, 1)
                chn_ids = torch.cat([chn_ids, last_row], dim=0)
                
            spec_masks = torch.cat([spec_masks, torch.ones(self.size - c, device=device)])
        spec_masks = spec_masks.bool()
        return dict(imgs=img, chn_ids=chn_ids, spec_masks=spec_masks, **add_kwargs)
    

# evaluation augmentation


class RandomHVFlip:
    def __init__(self, p=0.5):
        self.trf = transforms.Compose([
            transforms.RandomVerticalFlip(p=p),
            transforms.RandomHorizontalFlip(p=p),
        ])

    def __call__(self, data_obj):
        data_obj_out = {k: v for k,v in data_obj.items() if k != 'imgs'}
        data_obj_out['imgs'] = self.trf(data_obj['imgs'])
        return data_obj_out

class RandomResizedCrop:
    """ rrc for x_dict datatype with tensor in imgs key """
    def __init__(self, *args, **kwargs):
        self.transform = transforms.RandomResizedCrop(*args, **kwargs)

    def __call__(self, x_dict: dict):
        data_obj_out = {k: v for k,v in x_dict.items() if k != 'imgs'}
        data_obj_out['imgs'] = self.transform(x_dict['imgs'])
        return data_obj_out

class CenterCrop:
    def __init__(self, *args, **kwargs):
        self.transform = transforms.CenterCrop(*args, **kwargs)
    
    def __call__(self, x_dict: dict):
        data_obj_out = {k: v for k,v in x_dict.items() if k != 'imgs'}
        data_obj_out['imgs'] = self.transform(x_dict['imgs'])
        return data_obj_out

class ChnSelect:
    def __init__(self, idxs):
        idxs = [int(i) for i in idxs]
        self.idxs = idxs

    def __call__(self, x_dict):
        imgs = x_dict['imgs'][self.idxs]
        chn_ids = x_dict['chn_ids'][self.idxs]
        add_kwargs = {k:v for k,v in x_dict.items() if k not in ['imgs', 'chn_ids']}
        return dict(imgs=imgs, chn_ids=chn_ids, **add_kwargs)

class Resize:

    def __init__(self, *args, **kwargs):
        kwargs['antialias'] = kwargs.get('antialias', True)
        kwargs['size'] = self._parse_size(kwargs.get('size', None))
        self.resize = transforms.Resize(*args, **kwargs)

    def __call__(self, x_dict):
        img = x_dict['imgs']
        add_kwargs = {k:v for k,v in x_dict.items() if k not in ['imgs']}
        img = self.resize(img)
        return dict(imgs=img, **add_kwargs)

    def _parse_size(self, size): # so that inputs can be non 1:1 aspect ratio (like in fmow)
        """
        Parses the input size and returns it as a tuple of integers.

        Args:
            size (Union[str, int, Tuple[int, int]]): The input size, which can be a string, integer, or tuple.

        Returns:
            Tuple[int, int]: A tuple representing the size.
        """
        if isinstance(size, str):
            # Check if it's a tuple-like string
            if size.startswith('(') and size.endswith(')'):
                size = size.strip('()')
                size = tuple(map(int, size.split(',')))
            else:
                # Assume it's a single integer
                size = (int(size), int(size))
        elif isinstance(size, int):
            size = (size, size)
        elif isinstance(size, tuple) and len(size) == 2:
            size = tuple(size)
        else:
            raise ValueError(f"Invalid size format: {size}")

        return size


# used in appendix


class ChannelSimulator:
    def __init__(self,
                 wavelength_grid_size=1000, 
                 ):
        """
        Initialize the SpectralConvolutionTransform.

        Parameters:
        - valid_ranges: List of tuples [(min1, max1), (min2, max2), ...] specifying valid wavelength ranges -> moved one level up
        - wavelength_grid_size: Number of steps in the wavelength grid for numerical integration.
        - use_combined_srf_for_norm: If True, calculate normalization using combined SRF where source overlaps.
        - min_chanenls: this transform should only be applied to hyperspectral images with at least this many channels
        - max_tries: 
        """
        self.wavelength_grid_size = wavelength_grid_size

    def _gaussian_srf(self, wavelength, mean, std):
        """Calculate Gaussian SRF for a given wavelength."""
        return torch.exp(-0.5 * ((wavelength - mean) / std) ** 2)


    def spectral_convolution(self,
                        source_img,
                        target_srf_mean,
                        target_srf_std,
                        source_chn_ids=None,
                        wavelength_grid_size=1000):
        """
        Vectorized spectral convolution on a hyperspectral image using individual SRFs for each source channel.
        Produces a single channel image.

        Parameters:
        - source_img: Hyperspectral image with shape (C, H, W)
        - target_srf_mean: Mean wavelength of the target SRF
        - target_srf_std: Standard deviation of the target SRF
        - source_chn_ids: Source channel parameters (means, stds) shape (C, 2)
        - wavelength_grid_size: Number of points in wavelength grid

        Returns:
        - R: The resulting single-channel image with shape (H, W)
        """
        source_srf_means, source_srf_stds = source_chn_ids[:, 0], source_chn_ids[:, 1]

        # Set the integration range to ±3 stddevs from the target mean
        lambda_min = target_srf_mean - 3 * target_srf_std
        lambda_max = target_srf_mean + 3 * target_srf_std
        target_srf_mean = torch.tensor(target_srf_mean, device=source_img.device)
        target_srf_std = torch.tensor(target_srf_std, device=source_img.device)

        # Identify channels where the source SRF overlaps with the target SRF
        overlap_mask = ((source_srf_means + 3 * source_srf_stds) >= lambda_min) & \
                    ((source_srf_means - 3 * source_srf_stds) <= lambda_max)
        relevant_channels = torch.nonzero(overlap_mask).squeeze()

        # Define wavelength grid for integration
        wavelength_grid = torch.linspace(lambda_min, lambda_max, steps=wavelength_grid_size, 
                                    device=source_img.device)  # Shape: (W_grid,)

        # Calculate the target SRF values over the wavelength grid
        target_srf_values = self._gaussian_srf(wavelength_grid, target_srf_mean, target_srf_std)  # Shape: (W_grid,)

        # Calculate normalization factor
        combined_norm_factor = torch.trapz(target_srf_values, wavelength_grid)
        

        # Vectorized processing of all relevant channels
        # Calculate source SRFs for all relevant channels at once
        wavelength_grid_expanded = wavelength_grid[:, None]  # Shape: (W_grid, 1)
        source_srf_values = self._gaussian_srf(
            wavelength_grid_expanded,
            source_srf_means[relevant_channels],  # Shape: (n_relevant,)
            source_srf_stds[relevant_channels]    # Shape: (n_relevant,)
        )  # Shape: (W_grid, n_relevant)
        
        # Apply threshold
        source_srf_values = torch.where(source_srf_values > 1e-5, source_srf_values, 
                                    torch.zeros_like(source_srf_values))
        
        # Calculate combined SRF values for all channels at once
        combined_srf_values = source_srf_values * target_srf_values.unsqueeze(1)  # Shape: (W_grid, n_relevant)
        
        # Integrate for all channels simultaneously
        combined_srf_integrals = torch.trapz(combined_srf_values, wavelength_grid, dim=0)  # Shape: (n_relevant,)
        
        # Final multiplication and sum across channels
        R = (source_img[relevant_channels] * combined_srf_integrals.view(-1, 1, 1)).sum(dim=0)
        
        # Normalize
        R /= combined_norm_factor
        return R

    def _get_indices(self, data_obj):
        chn_indices = [(s,i) for s in range(len(data_obj['imgs'])) 
                    for i in range(len(data_obj['imgs'][s]))]
        return chn_indices

    def __call__(self, data_obj, target_ch_ids):
        """
        data_obj - dict with keys: 'img', 'chn_ids'
        target_ch_ids - tensor of shape (num_chns, 2) with the mean wavelength and standard deviation of each target channel SRF.
        """

        source_img = data_obj['imgs'][0]
        device = source_img.device

        if isinstance(target_ch_ids, torch.Tensor):
            target_srf_mean, target_srf_std = target_ch_ids[:,0], target_ch_ids[:,1]
        elif isinstance(target_ch_ids, list):
            target_srf_mean, target_srf_std = zip(*target_ch_ids)
            target_srf_mean, target_srf_std = torch.tensor(target_srf_mean, device=device).float(), torch.tensor(target_srf_std, device=device).float()
            # print(f'target_srf_mean: {target_srf_mean}, target_srf_std: {target_srf_std}')

        data_obj_out = {k: [] for k in data_obj.keys()}
        source_chn_ids = data_obj['chn_ids'][0]

        out_chns, out_chn_ids = [], []

        for mu, sigma in zip(target_srf_mean, target_srf_std):
            # print(f'Sampling target SRF: mean={target_srf_mean[i]}, std={target_srf_std[i]}')
            out_chns.append(self.spectral_convolution(
                            source_img=source_img,
                            target_srf_mean=mu,
                            target_srf_std=sigma,
                            source_chn_ids=source_chn_ids,
                            wavelength_grid_size=self.wavelength_grid_size
                            ))
        #stack to tensor
        out_chns = torch.stack(out_chns, dim=0)
        data_obj_out['imgs'] = [out_chns]
        out_chn_ids = torch.stack([target_srf_mean, target_srf_std], dim=1)
        data_obj_out['chn_ids'] = [out_chn_ids]

        return data_obj_out

class ChannelSimAugmentation():
    def __init__(self,
                n_channels_range: Tuple[int, int],
                valid_wavelength_ranges: List[Tuple[float, float]],
                valid_sigma_ranges: List[Tuple[float, float]],
                sigma_distribution: str = 'uniform',
                max_tries: int = 100,
                wavelength_grid_size=1000, 
                use_combined_srf_for_norm=False,
                min_channels=100,
                seed = 42,
                ):
        """
        Parameters:
        -----------
        n_channels_range : Tuple [int, int], e.g. [6,12]
            Number of channels to generate, pick randomly between these two values
        valid_wavelength_ranges : List[Tuple[float, float]]
            List of valid wavelength ranges, e.g. [(400,760), (890,1500)]
        valid_sigma_ranges : List[Tuple[float, float]]
            List of valid sigma ranges, e.g. [(5,50)]
        sigma_distribution : str
            Distribution to sample sigmas from ('uniform' or 'gaussian')
        max_tries : int
            Maximum number of attempts to generate valid parameters

        """

        self.wavelength_grid_size = wavelength_grid_size
        self.use_combined_srf_for_norm = use_combined_srf_for_norm
        self.max_tries = max_tries
        self.min_channels = min_channels
        self.n_channels_range = n_channels_range
        self.valid_wavelength_ranges = valid_wavelength_ranges
        self.valid_sigma_ranges = valid_sigma_ranges
        self.sigma_distribution = sigma_distribution

        self.chn_sim = ChannelSimulator(wavelength_grid_size=wavelength_grid_size)
        
        # self.generator = torch.Generator().manual_seed(seed)
        self.rng = np.random.default_rng(seed)

    def _get_indices(self, data_obj):
        chn_indices = [(s,i) for s in range(len(data_obj['imgs'])) 
                    for i in range(len(data_obj['imgs'][s]))]
        return chn_indices


    def _sample_sigma(self):
        """Sample sigma based on the specified distribution."""
        sigma_range = self.rng.choice(self.valid_sigma_ranges)
        if self.sigma_distribution == "uniform":
            return self.rng.uniform(*sigma_range)
        elif self.sigma_distribution == "gaussian":
            mean = (sigma_range[0] + sigma_range[1]) / 2
            stddev = (sigma_range[1] - sigma_range[0]) / 6  # 99.7% within range
            while True:
                sigma = self.rng.normal(mean, stddev)
                if sigma_range[0] <= sigma <= sigma_range[1]:
                    return sigma
        else:
            raise ValueError(f"Unsupported sigma distribution: {self.sigma_distribution}")

    def _filter_mu_ranges(self, sigma):
        """Filter valid mu ranges based on the width required by sigma."""
        required_width = 6 * sigma
        filtered_ranges = [
            (mu_min, mu_max)
            for mu_min, mu_max in self.valid_wavelength_ranges
            if mu_max - mu_min >= required_width
        ]
        return filtered_ranges

    def _sample_mu(self, filtered_ranges, sigma):
        """Sample mu from the filtered ranges."""
        while filtered_ranges:
            chosen_range = self.rng.choice(filtered_ranges)
            mu_min, mu_max = chosen_range
            mu = self.rng.uniform(mu_min + 3 * sigma, mu_max - 3 * sigma)
            return mu
        raise ValueError(f"No valid mu range found for sigma={sigma}. Please adjust input ranges.")

    def generate_srf_parameters(self, num_channels):
        """
        Generate (mu, sigma) tuples.

        Parameters:
        - num_channels (int): Number of (mu, sigma) pairs to generate.
        - max_tries (int): Maximum attempts to find valid (mu, sigma) pairs.

        Returns:
        - List of (mu, sigma) tuples.
        """
        params = []
        for _ in range(num_channels):
            for _ in range(self.max_tries):
                sigma = self._sample_sigma()
                filtered_ranges = self._filter_mu_ranges(sigma)
                if filtered_ranges:
                    mu = self._sample_mu(filtered_ranges, sigma)
                    params.append((mu, sigma))
                    break
            else:
                raise ValueError(f"Unable to generate valid (mu, sigma) pair after {self.max_tries} attempts.")
        return params

    def __call__(self, data_obj):
        chn_indices = self._get_indices(data_obj) #tuple of [(sensor, channels)]
        num_sensors = len(data_obj['imgs'])

        if num_sensors > 1:
            logger.warning( f"Only one sensor supported in channel:sim mode, provided {num_sensors}, using first sensor")

        if len(chn_indices) < self.min_channels:
            logger.error(f"ERROR: Number of channels in image is less than {self.min_channels}, skipping")
            return data_obj
        
        device = data_obj['imgs'][0].device
        #randomize the number of channels
        n_channels = self.rng.integers(self.n_channels_range[0],self.n_channels_range[1])
        
        # generate valid (mu,sigma) pairs in the valid wavelength ranges
        target_ch_ids = self.generate_srf_parameters(n_channels)

        #generate target SRFs:
        return self.chn_sim(data_obj, target_ch_ids)
