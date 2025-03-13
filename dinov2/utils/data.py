import os
import yaml
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import torch
from typing import List
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize
import random

def read_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def load_ds_cfg(ds_name):
    """ load chn_props and metainfo of dataset from file structure"""
    
    root = os.environ.get('CDIR', 'dinov2/configs/') # assumes current working directory in PanOpticOn/
    root = os.path.join(root, 'data')

    # get dataset
    dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d != 'satellites']
    ds = {}
    for d in dirs:
        for r, d, f in os.walk(os.path.join(root, d)):
            for file in f:
                if file[-5:] == '.yaml':
                    ds[file.split('.')[0]] = os.path.join(r, file)
    assert ds_name in ds, f'Dataset "{ds_name}" not found at {root} in folders {dirs}'
    ds_cfg = read_yaml(ds[ds_name])

    # get satellites
    sats = {}
    for r,d,f in os.walk(os.path.join(root, 'satellites')):
        for file in f:
            if file[-5:] == '.yaml':
                sats[file.split('.')[0]] = os.path.join(r, file) 

    # build chn_props
    chn_props = []
    sat_cfgs = {}
    for b in ds_cfg['bands']:
        sat_id, band_id = b['id'].split('/')
        if sat_id not in sat_cfgs:
            sat_cfgs[sat_id] = read_yaml(sats[sat_id])
        band_cfg = sat_cfgs[sat_id]['bands'][band_id]
        band_cfg['id'] = b['id']
        chn_props.append(band_cfg)
    metainfo = {k:v for k,v in ds_cfg.items() if k != 'bands'}
    return {'ds_name': ds_name, 'bands': chn_props, 'metainfo': metainfo}



def extract_wavemus(ds_cfg, return_sigmas=False):
    mus = [b['gaussian']['mu'] for b in ds_cfg['bands']]

    if not return_sigmas:
        return torch.tensor(mus, dtype=torch.int16)
    
    sigmas = [b['gaussian']['sigma'] for b in ds_cfg['bands']]
    return torch.tensor(list(zip(mus, sigmas)), dtype=torch.int16)

def dict_to_device(x_dict, device, keys: List = None, **kwargs):
    if keys is None:
        keys = x_dict.keys()
    for k in keys:
        x_dict[k] = x_dict[k].to(device, **kwargs)
    return x_dict

def plot_ds(*ds_names, log_gsd=True):
    if isinstance(ds_names[0], list):
        ds_names = ds_names[0]

    # load data
    out = []
    for ds_name in ds_names:
        ds_cfg = load_ds_cfg(ds_name)
        vals = [dict(
            ds=ds_name, mu=b['gaussian']['mu'], sat=b['id'].split('/')[0], gsd=b['GSD']) 
            for b in ds_cfg['bands']]
        out += vals
    df = pd.DataFrame(out)
    df['gsd_jitter'] = df['gsd'] + np.random.randn(len(df))*0.1

    # colored by satellite within dataset
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='mu', y='gsd_jitter', hue='ds', ax=ax)
    if log_gsd:
        ax.set_yscale('log')


def getimgsatl(id, dir, sensor, root_dir=None):

    if sensor == 's2':
        bands = ['tci', 'b05', 'b06', 'b07', 'b08', 'b11', 'b12']
        root_dir = os.path.join(root_dir, 'sentinel2')
    elif sensor == 'ls':
        bands = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11']
        root_dir = os.path.join(root_dir, 'landsat')
    else:
        raise NotImplementedError()

    sat_img = {}
    for b in bands:
        # img = read_image(os.path.join(root_dir, dir, b, f'{id}.png'))
        img = Image.open(os.path.join(root_dir, dir, b, f'{id}.png'))
        img = pil_to_tensor(img)
        sat_img[b] = img
    return sat_img

def plot_rgb(img):
    img = (img - img.min()) / (img.max() - img.min())
    plt.imshow(img.permute(1,2,0))
    plt.show()


    
def compute_dataset_stats(dataset, bsz=256, num_workers=2, subset=-1) -> None:
    """ 
    adjusted from https://github.com/stanfordmlgroup/USat/blob/main/usat/utils/helper.py
    Computes mean std for our datasets. Groups results by number of input channels.
    """

    # prepare dataset

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    if isinstance(subset, int) and subset > 0:
        subset = min(subset, len(dataset))
        indices = indices[:subset]
    elif isinstance(subset, float) and 0 < subset < 1: 
        indices = indices[:int(subset*len(dataset))]
    dataset = torch.utils.data.Subset(dataset, indices)

    def _list_resize(img_list):
        max_W = max([img.shape[-1] for img in img_list])
        max_H = max([img.shape[-2] for img in img_list])
        img_list = [resize(img, (max_H, max_W), antialias=True)
                    for img in img_list]
        return img_list
    
    class DatasetTensorOutput:
        def __init__(self, ds):
            self.ds = ds

        def __len__(self):
            return len(self.ds)
        
        def __getitem__(self, idx):
            view_list = self.ds[idx]
            if isinstance(view_list, tuple): # eval dataset
                x_dict = view_list[0]
                x_dict = dict(imgs=[x_dict['imgs']], chn_ids=x_dict['chn_ids'])
                view_list = [x_dict]
            assert len(view_list) == 1
            img_list = view_list[0]['imgs']
            return torch.cat(_list_resize(img_list))

    ds = DatasetTensorOutput(dataset)

    def collate_fn(l_batch):
        indices = {}
        for img in l_batch:
            nchns = img.size(0)
            if nchns not in indices:
                indices[nchns] = []
            indices[nchns].append(img)
        indices = {nchns: torch.stack(_list_resize(img_list)) for nchns, img_list in indices.items()}
        return [(nchns, img) for nchns,img in indices.items()]

    # actual iteration

    vals = {}
    dl = DataLoader(ds, batch_size=bsz, num_workers=num_workers, collate_fn=collate_fn)
    for data_list in tqdm(dl):
        for nchns, data in data_list:
            if not nchns in vals:
                vals[nchns] = dict(
                    c = 0,
                    s = torch.zeros(nchns),
                    ss = torch.zeros(nchns))

            vals[nchns]['s'] += torch.sum(data, (0,2,3))
            vals[nchns]['ss'] += torch.sum(data**2, (0,2,3))
            vals[nchns]['c'] += data.size(0) * data.size(2) * data.size(3)

    print(f"Dataset len: {len(dataset)}")
    for nchns, res in vals.items():
        mean = res['s'] / res['c']
        std = torch.sqrt((res['ss'] / res['c']) - (mean ** 2))
        print(f"nchns={nchns} with c={res['c']}, Mean: {mean}")
        print(f"nchns={nchns} with c={res['c']}, Std: {std}")