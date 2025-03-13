import torchvision.transforms.functional
from dinov2.utils.data import load_ds_cfg, extract_wavemus
from torch.utils.data import Dataset, Subset, IterableDataset
import pandas as pd
import numpy as np
import os
import torch
from torchvision import transforms
import logging
from torchvision.io import read_image

import h5py
import yaml
from tqdm import tqdm
import math
import torchvision
from dinov2.data.webdataset import _build_tars_work, build_tars, make_wds
import tifffile as tiff
import io
import time
import random

logger = logging.getLogger("dinov2")


class SatlasDataset(Dataset):

    # S2 from USAT, L9 calculated internally (10% subset), S1 from MMEarth: average of ascending and descending orbits

    MEAN = {
        's2': {
            'B04': 0.3475, 'B03': 0.3547, 'B02': 0.3804, 'B05': 0.3038, 'B06': 0.3843, 'B07': 0.4232, 'B08': 0.4157, 'B11': 0.3687, 'B12': 0.2847}, 
        'landsat': {
            'B01': 0.4089, 'B02': 0.3725, 'B03': 0.3479, 'B04': 0.3442, 'B05': 0.6330, 'B06': 0.5025, 'B07': 0.3665, 'B08': 0.3427, 'B09': 0.0702, 'B10': 0.9373, 'B11': 0.9399},
        's1': {
            'B01': 0.5356, 'B02': 0.4324} # vv and vh
    }
    STD = {
        's2': {
            'B04': 0.2394, 'B03': 0.1936, 'B02': 0.1836, 'B05': 0.1425, 'B06': 0.1434, 'B07': 0.1554, 'B08': 0.1526, 'B11': 0.1472, 'B12': 0.1264},
        'landsat': {
            'B01': 0.1701, 'B02': 0.1799, 'B03': 0.1923, 'B04': 0.2224, 'B05': 0.2728, 'B06': 0.2644, 'B07': 0.2348, 'B08': 0.2031, 'B09': 0.0415, 'B10': 0.1482, 'B11': 0.141},
        's1': {
            'B01': 0.0730, 'B02': 0.0780} # vv and vh
    }

    # load order of bands, correct order of bands in sensor s after loading is MEAN[s].keys()
    s2_bands_filenames = ['tci', 'b05', 'b06', 'b07', 'b08', 'b11', 'b12'] # tci saves the 3 bands B04,B03,B02 in one file
    s1_bands_filenames = ['vv', 'vh']
    landsat_bands_filenames = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11']
    
    local_to_yaml_map = {'s2' : 'satlas_s2',  'landsat': 'satlas_landsat', 's1': 'satlas_s1'}
    RGB_CHN_IDX = {
        's2': [0, 1, 2],
        'landsat': [3, 2, 1], 
    }


    def __init__(self, 
            root = None,
            num_sens:int = 2,
            normalize:bool = True, 
            metadata_path = None,
            keep_sensors = ['s1', 's2', 'landsat'],
            return_rgb = False,
            transform = None,
            full_spectra = False,
            hdf5_path = None,
            resize = False, # =True for debugging wds
        ):
        """

        :param num_sens: Number of sensors to use. Default is 2.
        :param normalize: Normalize the image. Default is True.
        :param root: Root directory of the dataset. Default is '${RESOURCE_DIR}/datasets'.
        :param keep_sensors: List of sensors to keep. Default is ['s1', 's2', 'landsat'].
        :param return_rgb: Return RGB image. Default is False.
        :param transform: Transform to apply to the image. Default is None.
        :param max_tries_load_img: Maximum number of tries to load the image. Default is 3.
        :param full_spectra: Use full spectral embedding. Default is False.
        """

        self.root = os.path.expandvars(root)
        self.M = num_sens
        self.normalize = normalize
        self.transform = transform
        self.return_rgb = return_rgb
        self.resize = resize
        assert all(sensor in self.local_to_yaml_map.keys() for sensor in keep_sensors), f'Invalid sensor name in {keep_sensors}'
        self.keep_sensors = keep_sensors

        # prepare hdf5
        self.hdf5_path = hdf5_path
        self.use_hdf5 = hdf5_path is not None
        if self.use_hdf5:
            self._img_read_fct = self._read_hdf5
        else:
            self._img_read_fct = self._read_file

        self.df = pd.read_parquet(metadata_path)
        self.chn_ids = {loc: extract_wavemus(load_ds_cfg(file), full_spectra) 
                        for loc,file in self.local_to_yaml_map.items()} 

        if self.normalize:
            logger.info('Building normalization transforms')
            self._build_ch_transforms()
        
    def _build_ch_transforms(self):
        channelwise_transforms = {}
        for sensor in self.MEAN.keys():
            if sensor not in channelwise_transforms:
                channelwise_transforms[sensor] = {}
            for band in self.MEAN[sensor].keys():
                channelwise_transforms[sensor][band] = transforms.Normalize(self.MEAN[sensor][band], self.STD[sensor][band])
        self.channelwise_transforms = channelwise_transforms

    def __len__(self):
        return len(self.df)

    def log_stats(self):
        sensor_counts = {sensor: 0 for sensor in self.sensor_name_mapping.keys()}
        for sensor in self.sensor_name_mapping.keys():
            sensor_counts[sensor] = self.df['sensor'].apply(lambda x: sensor in x).sum()
        logger.info(f'Dataset size: {self.__len__()}, sensor counts: {sensor_counts}')

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.use_hdf5 and not hasattr(self, "data_full") :
            self._open_hdf5(self.hdf5_path)

        def _get_idx():
            valid_sensor_idxs = [i for i, sensor in enumerate(row['sensor']) if sensor in self.keep_sensors]
            if len(valid_sensor_idxs) < self.M:
                raise ValueError(f'Not enough valid sensors in row {idx}. Found {len(valid_sensor_idxs)} sensors, expected >={self.M}')  
            
            idxs = np.random.choice(valid_sensor_idxs, self.M, replace=False)
            return idxs

        # load images

        views_list = []
        sensors = []
        views_idx = _get_idx()
        for i in views_idx:
            views_list.append(self._load_img(row, i))
            sensors.append(row['sensor'][i])

        # process image

        chn_ids_list  = [self.chn_ids[s] for s in sensors]
        chn_ids_list = [[t.unsqueeze(0) for t in tensor] for tensor in chn_ids_list]

        if self.normalize:
            for idx in range(len(views_list)):
                imgs_list = views_list[idx]
                sensor = sensors[idx]
                
                for j, band in enumerate(self.MEAN[sensor].keys()): # this is the correct band order!
                    imgs_list[j] = self.channelwise_transforms[sensor][band](imgs_list[j])
                        
                views_list[idx] = imgs_list       

        if self.return_rgb:
            for idx in range(len(views_idx)):
                sensor = sensors[idx]

                if sensor in ['s2', 'landsat']:
                    img_list = views_list[idx]
                    chn_ids = chn_ids_list[idx]
                    img_list = [torch.cat([imgs_list[i] for i in self.RGB_CHN_IDX[sensor]])]
                    chn_ids = [torch.cat([chn_ids[i] for i in self.RGB_CHN_IDX[sensor]])]
                    views_list[idx] = img_list
                    chn_ids_list[idx] = chn_ids

        if self.resize:
            for idx in range(len(views_list)):
                img_list = views_list[idx]
                chn_ids = chn_ids_list[idx]

                max_size = max([img.shape[-1] for img in img_list]) # H=W
                img_list = [torchvision.transforms.functional.resize(
                                img, max_size, antialias=True)
                            for img in img_list]
                img_list = [torch.cat(img_list)]
                chn_ids = [torch.cat(chn_ids)]

                views_list[idx] = img_list
                chn_ids_list[idx] = chn_ids

        out = [dict(imgs=i, chn_ids=c) for i,c in zip(views_list, chn_ids_list)]

        if self.transform:
            out = self.transform(out)

        return out
                    
    def _open_hdf5(self, path):
        self.data_full = h5py.File(path, "r")

    def _read_hdf5(self, rel_path):
        t = torch.tensor(self.data_full[rel_path])
        if t.ndim == 2:
            t = t.unsqueeze(0)
        elif t.ndim == 3:
            t = t.permute(2, 0, 1)
        return t

    def _read_file(self, rel_path):
        return read_image(os.path.join(self.root, rel_path))

    def _load_img(self, row, idx, normalize_to_0_1=True, ):
        #c onvert to [0-1] range because our normalizations params are in that range
        sensor = row['sensor'][idx]
        time = row['time'][idx]

        img_list = []
        if sensor == 's1' :
            for b in self.s1_bands_filenames:
                path = f'sentinel1/{time}/{b}/{row["id"]}.png'
                img = self._img_read_fct(path)
                # print(f'{path} shape: {img.shape}')
                if normalize_to_0_1:
                    img = img.type(torch.float) / 255.
                img_list.append(img)

        elif sensor == 'landsat':
            for b in self.landsat_bands_filenames:
                path = f'landsat/{time}/{b}/{row["id"]}.png'
                img = self._img_read_fct(path)
                if normalize_to_0_1:
                    img = img.type(torch.float) / 255.
                img_list.append(img)

        elif sensor == 's2':
            for band in self.s2_bands_filenames:

                path = f'sentinel2/{row["s2_dir"]}/{band}/{row["id"]}.png'
                img = self._img_read_fct(path)

                if img.shape[0] == 3:
                    for i in range(img.shape[0]):
                        if normalize_to_0_1:
                            img_list.append((img[i].type(torch.float) / 255.).unsqueeze(0)) 
                        else:
                            img_list.append(img[i].unsqueeze(0))
                else:
                    if normalize_to_0_1:
                        img = img.type(torch.float) / 255.
                    img_list.append(img)
                

        return img_list
        
    

######## Webdataset ########

WDS_USE_TIF = True

def build_tars_satl_webdataset(
        ds,
        out_base = None,
        dsname = 'dataset',
        overwrite = True,
        keep_sensors = ['s1', 's2', 'landsat'],
        **kwargs
    ):
    """ build wds tars for satlas dataset. Supports multiple processes with 
        kwargs dist_rank and dist_world_size. """
    
    # setup

    os.makedirs(out_base, exist_ok=True)
    metadata_file = os.path.join(out_base, f'{dsname}.yaml')
    if os.path.exists(metadata_file):
        if overwrite:
            logger.info('Overwriting existing dataset.')
        else:
            raise RuntimeError(
                f'{metadata_file} already exists. Set Overwrite=True to continue regardless.')

    handle = ds
    if isinstance(ds, Subset):
        handle = ds.dataset

    # metainfo & mean,std

    if kwargs.get('dist_rank',0) == 0:

        print(f'Use TIF: {WDS_USE_TIF}')
        # handle.df.to_csv(os.path.join(out_base, f'{dsname}.csv'))
        print('not writing csv')

        # get and save mean & std

        mean_std_dict = {}
        for sensor in keep_sensors:
            mean = [handle.MEAN_VALUES[f'{sensor}:{band}'] for band in 
                    handle.__getattribute__(f'{sensor}_bandname_map').keys()]
            std = [handle.STD_VALUES[f'{sensor}:{band}'] for band in 
                handle.__getattribute__(f'{sensor}_bandname_map').keys()]
            mean, std = torch.tensor(mean), torch.tensor(std)
            mean_std_dict[sensor] = (mean, std)
        torch.save(mean_std_dict, os.path.join(out_base, f'{dsname}_mean_std.pt'))

        # create metainfo file

        with open(metadata_file, 'a') as f:
            f.write('nsamples_per_shard:\n')

    # build tars

    sens_data = { # (idx, resize)
        's1': (0, 512), 
        's2': (1, 512),
        'landsat': (2, 256)}

    def write_to_tmp(idx, shard_base):
        name = handle.df.iloc[idx]['id']

        for s in keep_sensors:

            # load
            sens_idx, size = sens_data[s]
            img_list = handle._load_img(
                handle.df.iloc[idx], 
                sens_idx, 
                normalize_to_0_1 = not WDS_USE_TIF)
            if s == 'landsat':
                img_list = [
                    torchvision.transforms.functional.resize(img, size, antialias=True) 
                    for img in img_list]
            img = torch.cat(img_list)

            # store
            if WDS_USE_TIF:
                imgfile = os.path.join(shard_base, f'{name}.{s}.tif')
                with tiff.TiffWriter(imgfile) as tif:
                    # tif.save(img.numpy())
                    tif.write(img.numpy(), compression='zlib', compressionargs={'level': 9})
            else:
                imgfile = os.path.join(shard_base, f'{name}.{s}.npy')
                with open(imgfile, 'wb') as f:
                    np.save(f, img.numpy())

    _build_tars_work(ds, dsname, out_base, metadata_file, write_to_tmp, **kwargs)


class SatlasWds(IterableDataset):
    """ webdataset version of satlas """ 

    def __init__(
            self, 
            url, 
            normalize = True, 
            transform = None, 
            keep_sensors = ['s1','s2','landsat'], 
            num_views = 1,
            get_idx_mode = 'unique',
            full_spectra = False, 
            **kwargs):

        self.is_webdataset = True

        # get mean & std

        base_dir = os.path.dirname(url)
        pt_files = [f for f in os.listdir(base_dir) if f.endswith('.pt')]
        assert len(pt_files) == 1, f'Expected 1 .pt file, found {len(pt_files)}'
        mean_std_dict = torch.load(os.path.join(base_dir, pt_files[0]))
        have_sensors = list(mean_std_dict.keys())

        assert all(sensor in have_sensors for sensor in keep_sensors), f'have={have_sensors}, keep={keep_sensors}'
        if get_idx_mode == 'unique':
            assert num_views <= len(keep_sensors), f'num_views must be <= len(keep_sensors)'
        self.num_sens = len(keep_sensors)

        # norm, chn_ids, get_idx_fct 

        norm_trf = {s: transforms.Normalize(mean_std_dict[s][0], mean_std_dict[s][1]) 
                    for s in keep_sensors}

        m = {'s2' : 'satlas_s2',  'landsat': 'satlas_landsat', 's1': 'satlas_s1'}
        chn_ids_all = {k: extract_wavemus(load_ds_cfg(m[k]), full_spectra) for k in keep_sensors } 

        if get_idx_mode == 'unique':
            _get_idx_fct = lambda : np.random.choice(keep_sensors, num_views, replace=False)
        elif get_idx_mode == 'replace':
            _get_idx_fct = lambda : np.random.choice(keep_sensors, num_views, replace=True)
        else: 
            raise ValueError(f'Invalid get_idx_mode: {get_idx_mode}')

        # build wds

        def tiff_decoder(key, value):
            return tiff.imread(io.BytesIO(value))

        def make_sample(sample):

            # sample sensors 
            sensors = _get_idx_fct()

            # load sampled sensors
            imgs = []
            chn_ids = []
            for s in sensors:
                # print(sample['__key__'], s)

                if WDS_USE_TIF:
                    img = sample[f'{s}.tif']
                    img = torch.from_numpy(img) / 255.
                else:
                    img = torch.from_numpy(sample[f'{s}.npy'])
                if normalize:
                    img = norm_trf[s](img)
                imgs.append(img)
                chn_ids.append(chn_ids_all[s])

            out = [dict(imgs=[imgs[i]], chn_ids=[chn_ids[i]]) 
                   for i in range(num_views)] # list of views
            if transform:
                out = transform(out)
            
            return out
        
        decoder = tiff_decoder if WDS_USE_TIF else 'pil'
        self.ds = make_wds(url, make_sample, decoder=decoder, **kwargs)
        self.num_samples = self.ds.num_samples

    def __iter__(self):
        return iter(self.ds)


if __name__ == '__main__':
    import sys

    args = sys.argv[1:]

    out_base = args[0]
    dist_world_size = int(args[1])
    dist_rank = int(args[2])
    if dist_rank == 0:
        print(f'out_base: {out_base}, dist_world_size: {dist_world_size}, dist_rank: {dist_rank}')

    ds = SatlasDataset(
        root = os.path.join(os.environ['RDIR'], 'datasets/satlas'),
        metadata_path=os.path.join(os.environ['RDIR'], 'datasets/satlas/metadata_v2/fmow_iwm_onid_3sensors_all_clean.parquet'))

    indices = list(range(len(ds)))
    random.seed(21)
    random.shuffle(indices)
    ds = Subset(ds, indices)

    ######### args
    start = time.time()
    build_tars_satl_webdataset(
        ds=ds, 
        # nshards=5,
        nsamples_per_shard=770, # 800 or 770 for clean divisibility
        out_base=out_base,
        overwrite=True,
        dist_world_size=dist_world_size,
        dist_rank=dist_rank,
        rm_untarred_dirs=True
    )
    print(f'Done in {time.time() - start:.2f} s')

    ######### multiprocessing
    # import multiprocessing as mp
    # import time

    # nworkers = 2
    # pool = mp.Pool(nworkers)
    # dist_world_size = nworkers
    # print('Start pool ...')
    # start = time.time()
    # for w in range(nworkers):
    #     dist_rank = w
    #     pool.apply_async(build_tars_satl_webdataset, kwds = dict(
    #         ds=ds, 
    #         nshards=5,
    #         nsamples_per_shard=7, # 800 or 770 for clean divisibility
    #         out_base=out_base,
    #         overwrite=False,
    #         dist_world_size=dist_world_size,
    #         dist_rank=dist_rank,
    #         rm_untarred_dirs=False,
    #     ))
    # pool.close()
    # pool.join()
    # print(f'All workers done in {time.time() - start:.2f} s')

    
    ########### single process
    # build_tars_satl_webdataset(
    #     ds, 
    #     nshards=5,
    #     nsamples_per_shard=10, # 800
    #     out_base=out_base,
    #     overwrite=True,
    #     dist_world_size=dist_world_size,
    #     dist_rank=dist_rank,)

    # if dist_rank == 0:
    #     time.sleep(5)
    #     print('Sanity check')

    #     satl_wds = SatlasWds(
    #         url = os.path.join(out_base, 'dataset-*.tar'),
    #         resampled=False,
    #         normalize=True,
    #         subset = 1,
    #         num_views=3,
    #         keep_sensors=['s1','s2','landsat'])

    #     for x_dict in satl_wds:
    #         pass