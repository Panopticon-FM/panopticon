import os
import yaml
import glob
from dinov2 import distributed
from itertools import islice
import webdataset as wds
import random
import logging
import math
import time

logger = logging.getLogger("dinov2")


def _get_metainfo(url):
    base_dir = os.path.dirname(url)

    metadata_file = [f for f in os.listdir(base_dir) if f.endswith('.yaml')]
    assert len(metadata_file) == 1, f'Expected 1 .yaml file, found {len(metadata_file)}'
    metadata = yaml.load(open(os.path.join(base_dir, metadata_file[0]), 'r'), Loader=yaml.FullLoader)

    return metadata

def make_wds(url, make_sample, subset=-1, resampled=True, seed=42, decoder='pil'):
    """ creates a webdataset from url with our pattern"""

    logger.info(f'Loading webdataset from {url}')

    tarfiles = glob.glob(url)
    assert len(tarfiles) > 0, f'No tarfiles found in {url}'
    metadata = _get_metainfo(url)

    # subsetting

    def get_nsamples(tarfiles):
        return sum([metadata['nsamples_per_shard'][os.path.basename(tar)] for tar in tarfiles])
    nsamples_all = get_nsamples(tarfiles)
    ntarfiles_all = len(tarfiles)

    if subset != -1: # int or float subset

        if isinstance(subset, float):
            assert 0 < subset <= 1, f'float subset must be between 0 and 1, got {subset}'
            ntars = int(subset * len(tarfiles))
        elif isinstance(subset, int):
            assert 0 < subset <= len(tarfiles), f'int subset must be between 0 and number of tarfiles, got {subset}'
            ntars = subset

        random.seed(seed)
        random.shuffle(tarfiles)
        tarfiles = tarfiles[:ntars]

        nsamples = get_nsamples(tarfiles)

    else:
        nsamples = nsamples_all

    logger.info(f'Subset={subset}: {nsamples} ({nsamples/nsamples_all*100:.1f}%) samples, {len(tarfiles)}/{ntarfiles_all} shards')

    # adjust wds.split_by_node to use our own dinov2.distributed
    def split_by_node(src, group=None):
        rank = distributed.get_global_rank()
        world_size = distributed.get_global_size()

        if world_size > 1:
            yield from islice(src, rank, None, world_size)
        else:
            yield from src

    trainset = wds.WebDataset(
        tarfiles, 
        resampled=resampled, 
        # cache_dir=cache_dir, 
        nodesplitter=split_by_node)


    # trainset = trainset.shuffle(1000).decode("pil").map(make_sample)
    trainset = trainset.shuffle(1000).decode(decoder).map(make_sample)

    # add properties
    trainset.num_samples = nsamples # for calculating epochs 
    if not resampled:
        trainset = trainset.with_length(nsamples)

    return trainset


####### multiprocessing async parallel tar building (not working yet) #######

from tqdm import tqdm
import multiprocessing as mp
import itertools


def build_tars(
        ds, 
        dsname, 
        out_base, 
        metadata_file,
        write_to_tmp_fct, 
        nshards = math.inf, 
        nsamples_per_shard = 10,
        rm_untarred_dirs = False,
        nworkers = 1
    ):
    """ general build tar for wds function with multiprocessing """

    if nshards == math.inf:
        nshards = math.ceil(len(ds) / nsamples_per_shard)

    pool = mp.Pool(nworkers)
    print(f'Building tars with {nworkers} workers ...')
    for w in range(nworkers):

        pool.apply_async(_debug, args=(
            ds, 
            dsname, 
            out_base, 
            metadata_file, 
            write_to_tmp_fct, 
            nshards, 
            nsamples_per_shard, 
            rm_untarred_dirs,
            w,
            nworkers))
    
    pool.close()
    pool.join()
    print('Done')
    
def _debug(
        ds, 
        dsname, 
        out_base, 
        metadata_file,
        write_to_tmp_fct,
        shards,   
        nsamples_per_shard = 10,
        rm_untarred_dirs = False,
        display_prog_bar = False):
    print(dsname, shards)
    time.sleep(2)
    print('done')



def _build_tars_work(
        ds, 
        dsname, 
        out_base, 
        metadata_file,
        write_to_tmp_fct,

        nshards = math.inf,
        nsamples_per_shard = 10,
        rm_untarred_dirs = False,

        dist_rank = 0,
        dist_world_size = 1,
    ):
    """ worker function for build_tars """

    print(f'Building tars with rank {dist_rank}/{dist_world_size} ...')
    if nshards == math.inf:
        nshards = math.ceil(len(ds) / nsamples_per_shard)
    shards = list(itertools.islice(list(range(nshards)), dist_rank, nshards, dist_world_size))
    pbar = tqdm(total=len(shards) * nsamples_per_shard, disable=dist_rank!=0)

    for ishard in shards:

        shard_name = f'{dsname}-{ishard:06d}'
        shard_base = os.path.join(out_base, shard_name)
        os.makedirs(shard_base, exist_ok=True)

        # create files
        idx = ishard * nsamples_per_shard
        i_curr_shard = 0
        while i_curr_shard < nsamples_per_shard and idx < len(ds):
            
            write_to_tmp_fct(idx, shard_base)

            idx += 1
            i_curr_shard += 1   
            pbar.update(1) 

        if idx == len(ds) and i_curr_shard < nsamples_per_shard:
            print(f'Finished building {shard_name} early with {i_curr_shard} samples (pbar might be off)')

        # create tar
        time.sleep(2) # just to buffer any IO-write problems
        os.system(f'tar --sort=name -cf {os.path.join(out_base,shard_name)}.tar -C {out_base} {shard_name}')
        with open(metadata_file, 'a') as f:
            f.write(f'  {shard_name}.tar: {i_curr_shard}\n')

        if rm_untarred_dirs:
            os.system(f'rm -rf {shard_base}')

        ishard += 1

    pbar.close()