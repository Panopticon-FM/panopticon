import torch
from torch.utils.data import BatchSampler
import numpy as np
import random
from math import floor, ceil
from torch.utils.data import Dataset, IterableDataset, Subset
import itertools
import dinov2.distributed as distributed
import logging

logger = logging.getLogger("dinov2")


class CombinedDataset(Dataset):
    def __init__(self, datasets, bsz_list, pe_ids, transform=None):
        for d in datasets:
            assert not isinstance(d, IterableDataset)
        self.datasets = datasets       
        self.dataset_len = [len(ds) for ds in datasets]
        self.bsz_list = bsz_list
        self.pe_ids = pe_ids
        self.transform = transform
    
    def __getitem__(self, idx):
        dataset_idx, sample_idx = idx

        x = self.datasets[dataset_idx][sample_idx]
        if self.transform is not None:
            x = self.transform(x)
        return x
    
    def __len__(self):
        return sum(self.dataset_len)


class InfiniteCombinedBatchSampler:
    """ takes CombinedDataset and samples batches with different size from each ds """

    def __init__(self, dataset, drop_last=True, shuffle=True, seed=42, infinite=True):
        """ shuffle=True means shuffling batches, always shuffled within datasets """
        assert isinstance(dataset, CombinedDataset)
        assert drop_last, 'drop_last=False not implemented yet'

        self.drop_last = drop_last
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.infinite = infinite

        # extract infos from combined dataset

        self.dataset = dataset
        self.dataset_len = dataset.dataset_len
        self.n_datasets = len(self.dataset_len)

        bsz_list = dataset.bsz_list
        assert len(bsz_list) == self.n_datasets
        self.bsz_list = bsz_list

    def __len__(self):
        return sum(self._nbatches_rank())

    def _nbatches_rank(self):
        size, rank = self._get_distributed()
        nbatches_rank = [
            floor(l / b / size) for l,b in zip(self.dataset_len, self.bsz_list)] # drop_last = True
        return nbatches_rank

    def _get_distributed(self):
        size = distributed.get_global_size()
        rank = distributed.get_global_rank()
        return size, rank
    
    def _get_ds_sampler(self, iterable):
        size, rank = self._get_distributed()
        if self.shuffle:
            self._shuffle(iterable)
        return itertools.islice(iterable, rank, len(iterable), size)
    
    def _shuffle(self, iterable):
        self.rng.shuffle(iterable)
        self.rng = random.Random(self.rng.getrandbits(16))

    def _iter(self):

        samplers = []
        for i, ds in enumerate(self.dataset.datasets):
            samplers.append(self._get_ds_sampler(list(range(len(ds)))))

        nbatches_rank = self._nbatches_rank()
        batch_idx = [i for sublist in [[j]*n for j,n in enumerate(nbatches_rank)] for i in sublist]

        if self.shuffle:
            self._shuffle(batch_idx)

        for ds_idx in batch_idx:
            yield [(ds_idx, next(samplers[ds_idx])) for _ in range(self.bsz_list[ds_idx])] # advance samplers

    def __iter__(self):
        if self.infinite:
            while True:
                yield from self._iter()
        else:
            yield from self._iter()
    
 
############ more general combined dataset that also allows for IterableDataset

class CombinedIterableDataset(IterableDataset):
    """ Only needed if batch_sampler_mode=True. Else, can just use WdsMapWrapper 
        and ConcatDataset
    
        supports IterableDatasets, has built in batch sampler to yield exactly bsz 
        samples from each dataset 
        
        - the .num_samples attr is approximate if webdatasets are given"""

    def __init__(self, 
            datasets, 
            transform = None, 
            shuffle = True,

            # batch sampler args
            batch_sampler_mode = True, 
            bsz = None,
            infinite_sampler = False, 
            seed = 42,
            drop_last= True, 
            reload_wds_iters = False,
    ):
        
        self.datasets = datasets
        self.bsz = bsz
        self.transform = transform
        self.seed = seed
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.infinite_sampler = infinite_sampler    
        self.reload_wds_iters = reload_wds_iters
        logger.info(f'CombinedIterableDataset: batch_sampler_mode={batch_sampler_mode}, infinite_sampler={infinite_sampler}, drop_last={drop_last}, shuffle={shuffle}, seed={seed}, reload_wds_iters={reload_wds_iters}')

        if batch_sampler_mode:
            assert drop_last, 'not implemented yet (since less efficient during training & many edge cases)'
            assert bsz is not None, 'Batch size must be specified in batch sampler mode'
        self.batch_sampler_mode = batch_sampler_mode # if True, returns bsz samples from same dataset

        self.is_webdataset = []
        self.dataset_len = []
        for ds in self.datasets:
            while isinstance(ds, Subset):
                ds = ds.dataset
            if hasattr(ds, 'is_webdataset') and ds.is_webdataset:
                self.is_webdataset.append(True)
                self.dataset_len.append(ds.num_samples)
            elif isinstance(ds, IterableDataset):
                raise ValueError('non-webdataset IterableDatasets are not supported')
            else:
                self.is_webdataset.append(False)
                self.dataset_len.append(len(ds))
        self.num_samples = sum(self.dataset_len)

    def _start_iters(self):
        if hasattr(self, '_iters') and not self.reload_wds_iters:
            return
        self._iters = []
        for i, ds in enumerate(self.datasets):
            if self.is_webdataset[i]:
                self._iters.append(iter(ds))
            else:
                self._iters.append(ds)

    def _get_distributed(self):
        size = distributed.get_global_size()
        rank = distributed.get_global_rank()
        return size, rank

    def _shuffle(self, iterable):
        self.rng.shuffle(iterable)
        self.rng = random.Random(self.rng.getrandbits(32))

    def _get_ds_sampler(self, iterable):
        size, rank = self._get_distributed()
        if self.shuffle:
            self._shuffle(iterable)
        return itertools.islice(iterable, rank, len(iterable), size)

    def _iter(self):

        size, rank = self._get_distributed()
        self._start_iters()

        if self.batch_sampler_mode:

            _samplers = []
            for i, ds in enumerate(self.datasets):
                if self.is_webdataset[i]:
                    _samplers.append(None)
                else:
                    _samplers.append(self._get_ds_sampler(list(range(len(ds)))))

            def _get_sample(ds_idx):

                if self.is_webdataset[ds_idx]:
                    out = next(self._iters[ds_idx])
                else: 
                    idx = next(_samplers[ds_idx])
                    out = self._iters[ds_idx][idx]

                if self.transform is not None:
                    out = self.transform(out)
                return out

            nbatches_rank = [floor(l / self.bsz / size) for l in self.dataset_len] # drop_last = True
            _batch_idx = [i for sublist in [[j]*n for j,n in enumerate(nbatches_rank)] for i in sublist]

            if self.shuffle:
                self._shuffle(_batch_idx)

            for ds_idx in _batch_idx:
                for _ in range(self.bsz):
                    yield _get_sample(ds_idx)

        else: 

            samples = []
            for i, ds in enumerate(self.datasets):
                if self.is_webdataset[i]:
                    samples += [(i,) for _ in range(self.dataset_len[i])]
                else:
                    samples += [(i, j) for j in range(len(ds))]

            samples_rank = self._get_ds_sampler(samples)

            if self.shuffle:
                samples_rank = list(samples_rank)
                self._shuffle(samples_rank)

            for s in samples_rank:
                if self.is_webdataset[s[0]]:
                    out = next(iter(self.datasets[s[0]]))
                else:
                    out = self.datasets[s[0]][s[1]]
                if self.transform is not None:
                    out = self.transform(out)
                yield out

    def __iter__(self):
        if self.infinite_sampler:
            while True:
                yield from self._iter()
        else:
            yield from self._iter()


class WdsMapWrapper(Dataset):
    """ wraps our Webdataset to pretend to be a map-based one. Hacky solution,
        distribution in distributed env is completely handled by wds. 
        Infinite sampler also included in wds."""

    def __init__(self, wds, transform = None):
        assert wds.is_webdataset, 'wds must be a webdataset'
        # wds.with_length(wds.num_samples)
        self.wds = wds
        self.transform = transform
        self._iter = None

    def __len__(self):
        return self.wds.num_samples
    
    def __getitem__(self, idx):
        if self._iter is None:
            self._iter = iter(self.wds)
        out = next(self._iter)
        if self.transform is not None:
            out = self.transform(out)
        return out

        