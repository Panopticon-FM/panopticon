# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import json
import logging
import os
from typing import List, Optional

import torch
from torch.nn.functional import one_hot, softmax

import dinov2.distributed as distributed
from dinov2.data import SamplerType, make_data_loader
from dinov2.eval.metrics import build_metric
from dinov2.eval.utils import evaluate, extract_features, get_num_classes
import torch.distributed as dist

logger = logging.getLogger("dinov2")


class KnnModule(torch.nn.Module):
    """
    Gets knn of test features from all processes on a chunk of the train features

    Each rank gets a chunk of the train features as well as a chunk of the test features.
    In `compute_neighbors`, for each rank one after the other, its chunk of test features
    is sent to all devices, partial knns are computed with each chunk of train features
    then collated back on the original device.
    """

    def __init__(self, train_features, train_labels, nb_knn, T, device, num_classes=1000):
        super().__init__()

        self.use_dist = distributed.is_enabled()
        if self.use_dist:
            self.global_rank = distributed.get_global_rank()
            self.global_size = distributed.get_global_size()
        else:
            self.global_rank = 0
            self.global_size = 1

        self.device = device
        self.train_features_rank_T = train_features.chunk(self.global_size)[self.global_rank].T.to(self.device)
        self.candidates = train_labels.chunk(self.global_size)[self.global_rank].view(1, -1).to(self.device)

        self.nb_knn = nb_knn
        self.max_k = max(self.nb_knn)
        self.T = T
        self.num_classes = num_classes

    def _get_knn_sims_and_labels(self, similarity, train_labels):
        topk_sims, indices = similarity.topk(self.max_k, largest=True, sorted=True)
        neighbors_labels = torch.gather(train_labels, 1, indices)
        return topk_sims, neighbors_labels

    def _similarity_for_rank(self, features_rank, source_rank):
        # Send the features from `source_rank` to all ranks
        broadcast_shape = torch.tensor(features_rank.shape).to(self.device)
        if self.use_dist:
            dist.broadcast(broadcast_shape, source_rank)

        broadcasted = features_rank
        if self.use_dist:
            if self.global_rank != source_rank:
                broadcasted = torch.zeros(*broadcast_shape, dtype=features_rank.dtype, device=self.device)
            dist.broadcast(broadcasted, source_rank)

        # Compute the neighbors for `source_rank` among `train_features_rank_T`
        similarity_rank = torch.mm(broadcasted, self.train_features_rank_T)
        candidate_labels = self.candidates.expand(len(similarity_rank), -1)
        return self._get_knn_sims_and_labels(similarity_rank, candidate_labels)

    def _gather_all_knn_for_rank(self, topk_sims, neighbors_labels, target_rank):
        # Gather all neighbors for `target_rank`
        topk_sims_rank = retrieved_rank = None
        if self.global_rank == target_rank:
            topk_sims_rank = [torch.zeros_like(topk_sims) for _ in range(self.global_size)]
            retrieved_rank = [torch.zeros_like(neighbors_labels) for _ in range(self.global_size)]

        if self.use_dist:
            dist.gather(topk_sims, topk_sims_rank, dst=target_rank)
            dist.gather(neighbors_labels, retrieved_rank, dst=target_rank)
        else:
            topk_sims_rank = [topk_sims]
            retrieved_rank = [neighbors_labels]

        if self.global_rank == target_rank:
            # Perform a second top-k on the k * global_size retrieved neighbors
            topk_sims_rank = torch.cat(topk_sims_rank, dim=1)
            retrieved_rank = torch.cat(retrieved_rank, dim=1)
            results = self._get_knn_sims_and_labels(topk_sims_rank, retrieved_rank)
            return results
        return None

    def compute_neighbors(self, features_rank):
        for rank in range(self.global_size):
            topk_sims, neighbors_labels = self._similarity_for_rank(features_rank, rank)
            results = self._gather_all_knn_for_rank(topk_sims, neighbors_labels, rank)
            if results is not None:
                topk_sims_rank, neighbors_labels_rank = results
        return topk_sims_rank, neighbors_labels_rank

    def forward(self, features_rank):
        """
        Compute the results on all values of `self.nb_knn` neighbors from the full `self.max_k`
        """
        assert all(k <= self.max_k for k in self.nb_knn)

        topk_sims, neighbors_labels = self.compute_neighbors(features_rank)
        batch_size = neighbors_labels.shape[0]
        topk_sims_transform = softmax(topk_sims / self.T, 1)
        matmul = torch.mul(
            one_hot(neighbors_labels, num_classes=self.num_classes),
            topk_sims_transform.view(batch_size, -1, 1),
        )
        probas_for_k = {k: torch.sum(matmul[:, :k, :], 1) for k in self.nb_knn}
        return probas_for_k

class OneVsAllMultiKnn(torch.nn.Module):
    def __init__(self, train_features, train_labels, nb_knn, T, device, num_classes=1000):
        super().__init__()
        
        self.nb_knn = nb_knn
        self.max_k = max(self.nb_knn)
        self.T = T
        self.num_classes = num_classes
        self.device = device

        self.train_features = train_features
        self.train_labels = train_labels

    def forward(self, features):
        classes = torch.where(self.train_labels.sum(0) > 0)[0]
        probas_all = {k: torch.zeros(features.shape[0], self.num_classes) for k in self.nb_knn}
        for c in classes: 
            train_labels_class = self.train_labels[:, c]

            classifier = KnnModule(self.train_features, train_labels_class, self.nb_knn, self.T, self.device, num_classes=2)
            probas_for_k_per_class = classifier(features)
            for k in self.nb_knn:
                probas_all[k][:, c] = probas_for_k_per_class[k][:, 1]
        for k in self.nb_knn:
            probas_all[k] = softmax(probas_all[k], 1).cuda()
        logger.info(f'device: {probas_all[self.nb_knn[0]].device}')

        return probas_all


class DictKeysModule(torch.nn.Module):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def forward(self, features_dict, targets):
        for k in self.keys:
            features_dict = features_dict[k]
        return {"preds": features_dict, "target": targets}


def create_module_dict(*, module, n_per_class_list, n_tries, nb_knn, train_features, train_labels):
    modules = {}
    mapping = create_class_indices_mapping(train_labels)
    for npc in n_per_class_list:
        if npc < 0:  # Only one try needed when using the full data
            full_module = module(
                train_features=train_features,
                train_labels=train_labels,
                nb_knn=nb_knn,
            )
            modules["full"] = ModuleDictWithForward({"1": full_module})
            continue
        all_tries = {}
        for t in range(n_tries):
            final_indices = filter_train(mapping, npc, seed=t)
            k_list = list(set(nb_knn + [npc]))
            k_list = sorted([el for el in k_list if el <= npc])
            all_tries[str(t)] = module(
                train_features=train_features[final_indices],
                train_labels=train_labels[final_indices],
                nb_knn=k_list,
            )
        modules[f"{npc} per class"] = ModuleDictWithForward(all_tries)

    return ModuleDictWithForward(modules)


def filter_train(mapping, n_per_class, seed):
    torch.manual_seed(seed)
    final_indices = []
    for k in mapping.keys():
        index = torch.randperm(len(mapping[k]))[:n_per_class]
        final_indices.append(mapping[k][index])
    return torch.cat(final_indices).squeeze()


def create_class_indices_mapping(labels):
    unique_labels, inverse = torch.unique(labels, return_inverse=True)
    mapping = {unique_labels[i]: (inverse == i).nonzero() for i in range(len(unique_labels))}
    return mapping


class ModuleDictWithForward(torch.nn.ModuleDict):
    def forward(self, *args, **kwargs):
        return {k: module(*args, **kwargs) for k, module in self._modules.items()}


def eval_knn(
    model, # this already is a feature model
    train_dataset,
    val_dataset,
    metric_cfg,
    nb_knn,
    temperature,
    dl_cfg,
    gather_on_cpu,
    n_per_class_list=[-1],
    n_tries=1,
    is_multilabel=False,
):

    logger.info("Extracting features for train set...")
    train_features, train_labels = extract_features(
        model, train_dataset, gather_on_cpu=gather_on_cpu, dl_cfg=dl_cfg
    )

    val_dataloader = make_data_loader(
        dataset=val_dataset,
        sampler_type=SamplerType.DISTRIBUTED if distributed.is_enabled() else SamplerType.EPOCH,
        drop_last=False,
        shuffle=False,
        **dl_cfg
    )
    num_classes = get_num_classes(train_dataset)
    metric_collection = build_metric(metric_cfg, num_classes=num_classes)

    device = torch.cuda.current_device()
    knnmodule = OneVsAllMultiKnn if is_multilabel else KnnModule
    logger.info(f'Using knn module: {knnmodule} with num_classes {num_classes}')
    partial_module = partial(knnmodule, T=temperature, device=device, num_classes=num_classes)
    knn_module_dict = create_module_dict(
        module=partial_module,
        n_per_class_list=n_per_class_list,
        n_tries=n_tries,
        nb_knn=nb_knn,
        train_features=train_features,
        train_labels=train_labels,
    )
    postprocessors, metrics = {}, {}
    for n_per_class, knn_module in knn_module_dict.items():
        for t, knn_try in knn_module.items():
            postprocessors = {
                **postprocessors,
                **{(n_per_class, t, k): DictKeysModule([n_per_class, t, k]) for k in knn_try.nb_knn},
            }
            metrics = {**metrics, **{(n_per_class, t, k): metric_collection.clone() for k in knn_try.nb_knn}}
    model_with_knn = torch.nn.Sequential(model, knn_module_dict)

    # ============ evaluation ... ============
    logger.info("Start the k-NN classification.")
    _, results_dict = evaluate(model_with_knn, val_dataloader, postprocessors, metrics, device)
    logger.debug('Finished KNN classification')

    # Averaging the results over the n tries for each value of n_per_class
    for n_per_class, knn_module in knn_module_dict.items():
        first_try = list(knn_module.keys())[0]
        k_list = knn_module[first_try].nb_knn
        for k in k_list:
            keys = results_dict[(n_per_class, first_try, k)].keys()  # keys are e.g. `top-1` and `top-5`
            results_dict[(n_per_class, k)] = {
                key: torch.mean(torch.stack([results_dict[(n_per_class, t, k)][key] for t in knn_module.keys()]))
                for key in keys
            }
            for t in knn_module.keys():
                del results_dict[(n_per_class, t, k)]

    # output is (npc, t, k) where npc=number of samples per class, t=try, k=number of neighbors
    return results_dict


def eval_knn_with_model(
    model, # this already is a feature model
    output_dir,
    train_dataset,
    val_dataset,
    nb_knn=(10, 20, 100, 200),
    temperature=0.07,
    autocast_dtype=torch.float,
    metric_cfg=[{'id':'MulticlassAccuracy', 'top_k':1, 'average':'micro'}, {'id':'MulticlassAccuracy', 'top_k':5, 'average':'micro'}, {'id':'MulticlassAccuracy', 'top_k':5, 'average':'macro'}],
    gather_on_cpu=False,
    dl_cfg = {},
    n_per_class_list=[-1],
    n_tries=1,
    is_multilabel=False,
):

    autocast_dtype = model.autocast_dtype
    model.autocast_dtype = None # autocast is handled here
    with torch.cuda.amp.autocast(dtype=autocast_dtype):
        results_dict_knn = eval_knn(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            metric_cfg=metric_cfg,
            nb_knn=nb_knn,
            temperature=temperature,
            dl_cfg=dl_cfg,
            gather_on_cpu=gather_on_cpu,
            n_per_class_list=n_per_class_list,
            n_tries=n_tries,
            is_multilabel=is_multilabel,
        )


    results_list = []
    if distributed.is_main_process():

        # print all metrics
        dict_str = ''
        for knn_ in results_dict_knn.keys():
            dict_str += str(knn_) + ': {'
            for k, v in results_dict_knn[knn_].items():
                dict_str += f"{k}: {v.item() * 100.0:.2f}, "
            dict_str += '}\n'
        logger.info(f'All metrics result:\n{dict_str.strip()}')

        # save metrics
        metrics_file_path = os.path.join(output_dir, "results_eval_knn.json")
        with open(metrics_file_path, "a+") as f:
            for k, v in results_dict_knn.items():
                for kk,vv in v.items():
                    v[kk] = round(vv.item() * 100,2)
                f.write(json.dumps({str(k): str(v)}) + "\n")

        # add best classifier
        best_key = None
        best_val = 0
        for target_metric in build_metric(metric_cfg, num_classes=1000).keys():
            for k,v in results_dict_knn.items():
                if v[target_metric] > best_val:
                    best_key = k
                    best_val = v[target_metric]
            results_list.append(dict(
                best_classifier = best_key,
                value = best_val,
                metric = target_metric))

    if distributed.is_enabled():
        dist.barrier()
    return results_list