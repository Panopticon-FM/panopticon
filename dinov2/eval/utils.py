# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, Optional

import torch
from torch import nn
from torchmetrics import MetricCollection

from dinov2.utils.data import dict_to_device
from dinov2.data import DatasetWithEnumeratedTargets, SamplerType, make_data_loader
import dinov2.distributed as distributed
from dinov2.logging import MetricLogger
import torch.distributed as dist
import webdataset as wds
import numpy as np

logger = logging.getLogger("dinov2")


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
):
    model.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for samples, targets, *_ in metric_logger.log_every(data_loader, 10, header):
        outputs = model(dict_to_device(samples, device))
        targets = targets.to(device)

        if criterion is not None:
            loss = criterion(outputs, targets)
            metric_logger.update(loss=loss.item())

        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](outputs, targets)
            metric.update(**metric_inputs)

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")

    stats = {k: metric.compute() for k, metric in metrics.items()}
    logger.debug(f'Post compute')
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return metric_logger_stats, stats


def all_gather_and_flatten(tensor_rank):
    tensor_all_ranks = torch.empty(
        distributed.get_global_size(),
        *tensor_rank.shape,
        dtype=tensor_rank.dtype,
        device=tensor_rank.device,
    )
    tensor_list = list(tensor_all_ranks.unbind(0))
    if distributed.is_enabled():
        dist.all_gather(tensor_list, tensor_rank.contiguous())
    else:
        tensor_all_ranks = tensor_rank.unsqueeze(0)
    return tensor_all_ranks.flatten(end_dim=1)


def extract_features(model, dataset, dl_cfg, gather_on_cpu=False):
    dataset_with_enumerated_targets = DatasetWithEnumeratedTargets(dataset)
    sample_count = len(dataset_with_enumerated_targets)
    sampler_type = SamplerType.DISTRIBUTED if distributed.is_enabled() else SamplerType.EPOCH
    data_loader = make_data_loader(
        dataset=dataset_with_enumerated_targets,
        sampler_type=sampler_type,
        drop_last=False,
        shuffle=False,
        **dl_cfg
    )
    return extract_features_with_dataloader(model, data_loader, sample_count, gather_on_cpu)


@torch.inference_mode()
def extract_features_with_dataloader(model, data_loader, sample_count, gather_on_cpu=False):
    gather_device = torch.device("cpu") if gather_on_cpu else torch.device("cuda")
    metric_logger = MetricLogger(delimiter="  ")
    features, all_labels = None, None
    for samples, (index, labels_rank) in metric_logger.log_every(data_loader, 10):
        if isinstance(samples, dict):
            for k, v in samples.items():
                if isinstance(v, torch.Tensor):
                    samples[k] = v.cuda(non_blocking=True)
        else:
            samples = samples.cuda(non_blocking=True)
        labels_rank = labels_rank.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        features_rank = model(samples).float()

        # init storage feature matrix
        if features is None:
            features = torch.zeros(sample_count, features_rank.shape[-1], device=gather_device)
            labels_shape = list(labels_rank.shape)
            labels_shape[0] = sample_count
            all_labels = torch.full(labels_shape, fill_value=-1, device=gather_device)
            logger.info(f"Storing features into tensor of shape {features.shape}")

        # share indexes, features and labels between processes
        index_all = all_gather_and_flatten(index).to(gather_device)
        features_all_ranks = all_gather_and_flatten(features_rank).to(gather_device)
        labels_all_ranks = all_gather_and_flatten(labels_rank).to(gather_device)

        # update storage feature matrix
        if len(index_all) > 0:
            features.index_copy_(0, index_all, features_all_ranks)
            all_labels.index_copy_(0, index_all, labels_all_ranks)

    logger.info(f"Features shape: {tuple(features.shape)}")
    logger.info(f"Labels shape: {tuple(all_labels.shape)}")

    assert torch.all(all_labels > -1)

    return features, all_labels

def get_num_classes(ds):
    if isinstance(ds, wds.WebDataset):
        num_classes = ds.num_classes  # assigned in make_dataset
    else:
        if isinstance(ds, torch.utils.data.Subset):
            ds = ds.dataset
        # num_classes = len(torch.unique(torch.Tensor(handle.get_targets().astype(int))))
        num_classes = ds.num_classes
    return num_classes

def sample_indices_hs(
        nbands,
        nsamples,
        exclude = [5,15,29,47,53,58,64,69,74,89,148,193], # sentinel2 indices of corine
        radius_around_excluded = 3,
        mode = 'bin_before_remove'
    ):

    indices_to_exclude = set()
    for i in exclude:
        indices_to_exclude.update(range(i-radius_around_excluded, i+radius_around_excluded+1))
    
    if mode == 'remove_before_bin':
        indices = np.array([i for i in range(nbands) if i not in indices_to_exclude])
        parts = np.array_split(indices, nsamples)
    elif mode == 'bin_before_remove':
        parts = np.array_split(np.array(range(nbands)), nsamples)
        parts = [np.array([i for i in part if i not in indices_to_exclude]) for part in parts]
    else:
        raise ValueError(f'Unknown mode {mode}')
    
    indices_out = []
    for i, part in enumerate(parts):
        assert len(part) > 0, f'part {i} is empty, need to decrease radius, change mode, or decrease nsamples '
        print(f'part {i}: {part[0]}-{part[-1]}')
        idx = np.random.choice(part, 1)
        print(f'picked {idx}')
        indices_out.append(idx)
    
    return np.concatenate(indices_out)