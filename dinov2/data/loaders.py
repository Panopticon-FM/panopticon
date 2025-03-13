# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from dotenv import load_dotenv

from dinov2.data.combined_ds import CombinedDataset, InfiniteCombinedBatchSampler, WdsMapWrapper, CombinedIterableDataset


load_dotenv()
import logging
from enum import Enum
from typing import Any, Callable, Iterable, List, Optional, TypeVar

import torch
from torch.utils.data import Sampler, ConcatDataset

from dinov2.data.datasets.benv2 import BigEarthNetv2Wrapper, create_benv2_wds
from dinov2.data.datasets.geobench import GeobenchDataset
from dinov2.data.datasets.spectral_earth import SpectralEarthDataset
from dinov2.data.datasets.mmearth import MMEarthWrapper
from dinov2.data.datasets.satlas import SatlasDataset, SatlasWds
from dinov2.data.datasets.eurosat_sar import EurosatSAR
from dinov2.data.datasets.resisc45 import RESISC45
from dinov2.data.augmentations import make_augmentation
from torch.utils.data import Subset
from dinov2.data.datasets.fmow_original import FmowDatasetOriginal
from .datasets import DummyDataset, FmowDataset
from .samplers import EpochSampler, InfiniteSampler, ShardedInfiniteSampler
from copy import deepcopy

import webdataset as wds

import random
import math




logger = logging.getLogger("dinov2")

class SamplerType(Enum):
    DISTRIBUTED = 0
    EPOCH = 1
    INFINITE = 2
    SHARDED_INFINITE = 3
    SHARDED_INFINITE_NEW = 4


class ConcatDatasetTrf(ConcatDataset):
    def __init__(self, datasets, transform=None):
        super().__init__(datasets)
        self.transform = transform

    def __getitem__(self, idx):
        x = super().__getitem__(idx)
        if self.transform is not None:
            x = self.transform(x)
        return x

def make_dataset(cfg, pretrain_augm_cfg=None, seed=42, resampled=True):
    """ Main function to build datasets, resampled only used for eval wds """

    cfg = deepcopy(cfg)
    pretrain_augm_cfg = deepcopy(pretrain_augm_cfg)
    id = cfg.pop('id')
    transform_cfg = cfg.pop('transform', [])
    subset = cfg.pop('subset', -1)
    pretrain_augm_overwrites = cfg.pop('pretrain_augm_overwrites', {})

    logger.info(f'Building dataset "{id}" ...')

    # build transform
    if pretrain_augm_cfg is not None:
        pretrain_augm_cfg.update(pretrain_augm_overwrites)
        transform_cfg.append(pretrain_augm_cfg)
    transform = make_augmentation(transform_cfg)

    # build datasets for train
    if id == 'DummyDataset':
        ds = DummyDataset(**cfg, transform=transform)
        ds_modalities_out = 1

    elif id == 'FmowDataset':
        ds = FmowDataset(**cfg, transform=transform)
        ds_modalities_out = ds.M

    elif id == 'MMEarth':
        ds = MMEarthWrapper(**cfg, transform=transform)
        ds_modalities_out = len(ds.modalities)
        if 'month' in ds.modalities:
            ds_modalities_out -= 1

    elif id == 'SpectralEarth':
        ds = SpectralEarthDataset(**cfg, transform=transform)
        ds_modalities_out = 1

    elif id == 'SatlasDataset':
        ds = SatlasDataset(**cfg, transform=transform)
        ds_modalities_out = ds.M

    elif id == 'satl_webdataset':
        ds = SatlasWds(**cfg, transform=transform)
        ds_modalities_out = ds.num_sens

    elif id == 'ConcatDataset': # shuffle samples from all datasets
        assert len(transform_cfg) <= 1, 'No additional transforms supported for ConcatDataset yet'
        datasets = cfg.pop('datasets')
        datasets = [make_dataset(d, seed=seed, pretrain_augm_cfg=pretrain_augm_cfg) for d in datasets]
        ds = ConcatDataset(datasets)

    elif id == 'CombinedDataset': # shuffle batches from all datasets
        assert len(transform_cfg) == 1, 'No additional transforms supported for CombinedDataset yet'
        ds_cfg_list = cfg.pop('datasets')
        bsz_list = [d.pop('bsz') for d in ds_cfg_list]
        pe_ids = [d.pop('pe_id', d['id']) for d in ds_cfg_list]
        datasets = [make_dataset(d, seed=seed, pretrain_augm_cfg=pretrain_augm_cfg) for d in ds_cfg_list]
        ds = CombinedDataset(datasets, bsz_list=bsz_list, pe_ids=pe_ids)
        assert subset <= 0, 'Subset not supported for CombinedDataset'

    elif id == 'CombinedIterableDataset':
        assert len(transform_cfg) == 1, 'No additional transforms supported for CombinedIterableDataset yet'
        ds_cfg_list = cfg.pop('datasets')
        datasets = [make_dataset(d, seed=seed, pretrain_augm_cfg=pretrain_augm_cfg) for d in ds_cfg_list]
        ds = CombinedIterableDataset(datasets, **cfg)
        assert subset <= 0, 'Subset not supported for CombinedDataset'

    elif id == 'WdsMapWrapper':
        wds_cfg = cfg.pop('wds')
        wds = make_dataset(wds_cfg, seed=seed)
        ds = WdsMapWrapper(**cfg, wds=wds, transform=transform)

    # build datasets for eval
    elif 'geobench' in id:
        ds = GeobenchDataset(ds_name=id.split('.')[1], **cfg, transform=transform)

    elif id == 'benv2':
        ds = BigEarthNetv2Wrapper(**cfg, transform=transform)

    elif id == 'benv2_webdataset':
        ds = create_benv2_wds(**cfg, 
                                   transform=transform, 
                                   resampled=resampled, 
                                   subset=subset,
                                   seed=seed)

    elif id == 'eurosat-sar':
        ds = EurosatSAR(**cfg, transform=transform)

    elif id == 'resisc45':
        ds = RESISC45(**cfg, transform=transform)

    elif id == 'fmow':
        ds = FmowDatasetOriginal(**cfg, transform=transform)

    else:
        raise ValueError(f'Unsupported dataset "{id}"')

    if not isinstance(ds, torch.utils.data.IterableDataset):
        logger.info(f'Built dataset "{id}" with #samples {len(ds)}')
    else:
        logger.info(f'Built dataset "{id}"')

    # subset

    if subset > 0 and not isinstance(ds, torch.utils.data.IterableDataset):
        # logger.warn('Only for checking variance in seed!')
        # seed = seed + random.randint(0, 1000) 

        def sample_indices(n, k):
            generator = torch.Generator().manual_seed(seed)
            return torch.multinomial(torch.ones(n) / n, k, replacement=False, generator=generator).tolist()

        if isinstance(subset, float):
            assert 0.0 < subset <= 1.0, 'Float subset must be in range (0, 1].'
            if subset < 1.0:
                subset_indices = sample_indices(len(ds), int(len(ds)*subset))
                ds = Subset(ds, subset_indices)
        elif isinstance(subset, int):
            assert subset > 0, 'Int subset must be greater than 0.'
            if subset < len(ds):
                subset_indices = sample_indices(len(ds), subset)
                ds = Subset(ds, subset_indices)
            else:
                sampler = EpochSampler(size=subset, sample_count=len(ds), shuffle=False, seed=seed)
                subset_indices = sampler._get_iterable().tolist()
                ds = Subset(ds, subset_indices)
        else:
            raise ValueError(f'Unsupported subset type "{type(subset)}"')
        logger.info(f'Got subset={subset}, subsampled dataset to #samples {len(ds)} ')
    
    if not hasattr(ds,'is_webdataset'):
        ds.is_webdataset = False
    return ds


def _make_sampler(
    *,
    dataset,
    type: Optional[SamplerType] = None,
    shuffle: bool = False,
    seed: int = 0,
    size: int = -1,
    advance: int = 0,
    drop_last: bool = False
) -> Optional[Sampler]:
    sample_count = len(dataset)

    if type == SamplerType.INFINITE:
        logger.info("sampler: infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        return InfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
        )
    elif type in (SamplerType.SHARDED_INFINITE, SamplerType.SHARDED_INFINITE_NEW):
        logger.info("sampler: sharded infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        # TODO: Remove support for old shuffling
        use_new_shuffle_tensor_slice = type == SamplerType.SHARDED_INFINITE_NEW
        return ShardedInfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
            use_new_shuffle_tensor_slice=use_new_shuffle_tensor_slice,
        )
    elif type == SamplerType.EPOCH:
        logger.info("sampler: epoch")
        if advance > 0:
            raise NotImplementedError("sampler advance > 0 is not supported")
        size = size if size > 0 else sample_count
        logger.info(f"# of samples / epoch: {size:,d}")
        return EpochSampler(
            size=size,
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
        )
    elif type == SamplerType.DISTRIBUTED:
        logger.info("sampler: distributed")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        if advance > 0:
            raise ValueError("sampler advance > 0 is invalid")
        return torch.utils.data.DistributedSampler(
            dataset=dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    logger.info("sampler: none")
    return None


T = TypeVar("T")


def make_data_loader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 0,
    sampler_type: Optional[SamplerType] = SamplerType.INFINITE,
    sampler_size: int = -1,
    sampler_advance: int = 0,
    drop_last: bool = False,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable[[List[T]], Any]] = None,
):
    """
    Creates a data loader with the specified parameters.

    Args:
        dataset: A dataset (third party, LaViDa or WebDataset).
        batch_size: The size of batches to generate.
        num_workers: The number of workers to use.
        shuffle: Whether to shuffle samples.
        seed: The random seed to use.
        sampler_type: Which sampler to use: EPOCH, INFINITE, SHARDED_INFINITE, SHARDED_INFINITE_NEW, DISTRIBUTED or None.
        sampler_size: The number of images per epoch (when applicable) or -1 for the entire dataset.
        sampler_advance: How many samples to skip (when applicable).
        drop_last: Whether the last non-full batch of data should be dropped.
        persistent_workers: maintain the workers Dataset instances alive after a dataset has been consumed once.
        collate_fn: Function that performs batch collation
    """

    if hasattr(dataset, 'is_webdataset') and dataset.is_webdataset:
        logger.info("Detected WebDataset. Using DataLoader with WebLoader.")

        dl = wds.WebLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,)

        if hasattr(dataset, '__len__'):
            dl = dl.with_length(math.ceil(len(dataset) / batch_size))

        return dl


    if isinstance(dataset, CombinedDataset):
        logger.info("Detected CombinedDataset. Using InfiniteCombinedBatchSampler.")

        batch_sampler = InfiniteCombinedBatchSampler(
            dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            infinite=True)

        bsz_kwargs = dict(
            batch_sampler=batch_sampler)

    else:
        logger.info(f"Detected non-CombinedDataset. Using {sampler_type} with bsz={batch_size}.")

        sampler = _make_sampler(
            dataset=dataset,
            type=sampler_type,
            shuffle=shuffle,
            seed=seed,
            size=sampler_size,
            advance=sampler_advance,
            drop_last=drop_last)
        
        bsz_kwargs = dict(
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last)

    logger.info(f"DataLoader kwargs: num_workers={num_workers}, pin_memory={pin_memory}, drop_last={drop_last}, persistent_workers={persistent_workers}")
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        **bsz_kwargs,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    try:
        logger.info(f"# of batches: {len(data_loader):,d}")
    except TypeError:  # data loader has no length
        logger.info("infinite data loader")
    return data_loader
