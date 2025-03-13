# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""BigEarthNetv2 dataset."""

import glob
import os
from typing import Callable, Optional, List

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.figure import Figure
from rasterio.enums import Resampling
from torch import Tensor
import pandas as pd
from torchvision import transforms

from torch.utils.data import Dataset, Subset
import logging
from pathlib import Path

from dinov2.utils.data import load_ds_cfg, extract_wavemus
from datetime import datetime

import yaml
from tqdm import tqdm
import random
import math
from dinov2.data.webdataset import make_wds, _get_metainfo

logger = logging.getLogger('dinov2')


def sort_sentinel2_bands(x: Path) -> str:
    """Sort Sentinel-2 band files in the correct order."""
    x = os.path.basename(x).split('_')[-1]
    x = os.path.splitext(x)[0]
    if x == 'B8A':
        x = 'B08A'
    return x

class BigEarthNetv2(Dataset):
    """BigEarthNet dataset.

    The `BigEarthNet <https://bigearth.net/>`__
    dataset is a dataset for multilabel remote sensing image scene classification.

    Dataset features:

    * 590,326 patches from 125 Sentinel-1 and Sentinel-2 tiles
    * Imagery from tiles in Europe between Jun 2017 - May 2018
    * 12 spectral bands with 10-60 m per pixel resolution (base 120x120 px)
    * 2 synthetic aperture radar bands (120x120 px)
    * 43 or 19 scene classes from the 2018 CORINE Land Cover database (CLC 2018)

    Dataset format:

    * images are composed of multiple single channel geotiffs
    * labels are multiclass, stored in a single json file per image
    * mapping of Sentinel-1 to Sentinel-2 patches are within Sentinel-1 json files
    * Sentinel-1 bands: (VV, VH)
    * Sentinel-2 bands: (B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12)
    * All bands: (VV, VH, B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12)
    * Sentinel-2 bands are of different spatial resolutions and upsampled to 10m

    Dataset classes (43):

    0. Continuous urban fabric
    1. Discontinuous urban fabric
    2. Industrial or commercial units
    3. Road and rail networks and associated land
    4. Port areas
    5. Airports
    6. Mineral extraction sites
    7. Dump sites
    8. Construction sites
    9. Green urban areas
    10. Sport and leisure facilities
    11. Non-irrigated arable land
    12. Permanently irrigated land
    13. Rice fields
    14. Vineyards
    15. Fruit trees and berry plantations
    16. Olive groves
    17. Pastures
    18. Annual crops associated with permanent crops
    19. Complex cultivation patterns
    20. Land principally occupied by agriculture, with significant
        areas of natural vegetation
    21. Agro-forestry areas
    22. Broad-leaved forest
    23. Coniferous forest
    24. Mixed forest
    25. Natural grassland
    26. Moors and heathland
    27. Sclerophyllous vegetation
    28. Transitional woodland/shrub
    29. Beaches, dunes, sands
    30. Bare rock
    31. Sparsely vegetated areas
    32. Burnt areas
    33. Inland marshes
    34. Peatbogs
    35. Salt marshes
    36. Salines
    37. Intertidal flats
    38. Water courses
    39. Water bodies
    40. Coastal lagoons
    41. Estuaries
    42. Sea and ocean

    Dataset classes (19):

    0. Urban fabric
    1. Industrial or commercial units
    2. Arable land
    3. Permanent crops
    4. Pastures
    5. Complex cultivation patterns
    6. Land principally occupied by agriculture, with significant
       areas of natural vegetation
    7. Agro-forestry areas
    8. Broad-leaved forest
    9. Coniferous forest
    10. Mixed forest
    11. Natural grassland and sparsely vegetated areas
    12. Moors, heathland and sclerophyllous vegetation
    13. Transitional woodland, shrub
    14. Beaches, dunes, sands
    15. Inland wetlands
    16. Coastal wetlands
    17. Inland waters
    18. Marine waters

    The source for the above dataset classes, their respective ordering, and
    43-to-19-class mappings can be found here:

    * https://git.tu-berlin.de/rsim/BigEarthNet-S2_19-classes_models/-/blob/master/label_indices.json

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/IGARSS.2019.8900532

    """  # noqa: E501

    class_sets = {
        19: [
            "Urban fabric",
            "Industrial or commercial units",
            "Arable land",
            "Permanent crops",
            "Pastures",
            "Complex cultivation patterns",
            "Land principally occupied by agriculture, with significant areas of"
            " natural vegetation",
            "Agro-forestry areas",
            "Broad-leaved forest",
            "Coniferous forest",
            "Mixed forest",
            "Natural grassland and sparsely vegetated areas",
            "Moors, heathland and sclerophyllous vegetation",
            "Transitional woodland, shrub",
            "Beaches, dunes, sands",
            "Inland wetlands",
            "Coastal wetlands",
            "Inland waters",
            "Marine waters",
        ],
        43: [
            "Continuous urban fabric",
            "Discontinuous urban fabric",
            "Industrial or commercial units",
            "Road and rail networks and associated land",
            "Port areas",
            "Airports",
            "Mineral extraction sites",
            "Dump sites",
            "Construction sites",
            "Green urban areas",
            "Sport and leisure facilities",
            "Non-irrigated arable land",
            "Permanently irrigated land",
            "Rice fields",
            "Vineyards",
            "Fruit trees and berry plantations",
            "Olive groves",
            "Pastures",
            "Annual crops associated with permanent crops",
            "Complex cultivation patterns",
            "Land principally occupied by agriculture, with significant areas of"
            " natural vegetation",
            "Agro-forestry areas",
            "Broad-leaved forest",
            "Coniferous forest",
            "Mixed forest",
            "Natural grassland",
            "Moors and heathland",
            "Sclerophyllous vegetation",
            "Transitional woodland/shrub",
            "Beaches, dunes, sands",
            "Bare rock",
            "Sparsely vegetated areas",
            "Burnt areas",
            "Inland marshes",
            "Peatbogs",
            "Salt marshes",
            "Salines",
            "Intertidal flats",
            "Water courses",
            "Water bodies",
            "Coastal lagoons",
            "Estuaries",
            "Sea and ocean",
        ],
    }

    label_converter = { #TODO: convert to latest table
        0: 0,
        1: 0,
        2: 1,
        11: 2,
        12: 2,
        13: 2,
        14: 3,
        15: 3,
        16: 3,
        18: 3,
        17: 4,
        19: 5,
        20: 6,
        21: 7,
        22: 8,
        23: 9,
        24: 10,
        25: 11,
        31: 11,
        26: 12,
        27: 12,
        28: 13,
        29: 14,
        33: 15,
        34: 15,
        35: 16,
        36: 16,
        38: 17,
        39: 17,
        40: 18,
        41: 18,
        42: 18,
    }

    splits_metadata = {
        "train": {
            "filename": "metadata.parquet",
        },
        "validation": {
            "filename": "metadata.parquet",
        },
        "test": {
            "filename": "metadata.parquet",
        },
    }
    metadata_locs = {
        "s1": {
            "directory": "BigEarthNet-S1",
        },
        "s2": {
            "directory": "BigEarthNet-S2",
        },
        "maps": {
            "directory": "Reference_Maps",
        },
    }
    image_size = (120, 120)

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: str = "all",
        num_classes: int = 19,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new BigEarthNet dataset instance.

        Args:
            root: root directory where dataset can be found
            split: train/val/test split to load
            bands: load Sentinel-1 bands, Sentinel-2, or both. one of {s1, s2, all}
            num_classes: number of classes to load in target. one of {19, 43}
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        assert split in self.splits_metadata
        assert bands in ["s1", "s2", "all"]
        assert num_classes in [43, 19]
        self.root = root
        self.split = split
        self.bands = bands
        self.num_classes = num_classes
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.class2idx_43 = {c: i for i, c in enumerate(self.class_sets[43])}
        self.class2idx_19 = {c: i for i, c in enumerate(self.class_sets[19])} 
        self.folders = self._load_folders()

    def class2idx(self, label:str, level=19):
        assert level == 19 or level == 43, "level must be 19 or 43"
        return self.class2idx_19[label] if level == 19 else self.class2idx_43[label]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image = self._load_image(index)
        label = self._load_target(index)
        sample: dict[str, Tensor] = {"image": image, "label": label}
        # sample: dict[str, Tensor] = {"image": image, "label": None}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.folders)

    def _load_folders(self) -> list[dict[str, str]]:
        """Load folder paths.

        Returns:
            list of dicts of s1 and s2 folder paths
        """
        filename = self.splits_metadata[self.split]["filename"]
        dir_s1 = self.metadata_locs["s1"]["directory"]
        dir_s2 = self.metadata_locs["s2"]["directory"]
        dir_maps = self.metadata_locs["maps"]["directory"]

        self.metadata = pd.read_parquet(os.path.join(self.root, filename))
        self.metadata = self.metadata[self.metadata["split"] == self.split].reset_index(drop=True)

        def construct_folder_path(root, dir, patch_id, remove_last:int=2):
            tile_id = '_'.join(patch_id.split('_')[:-remove_last])
            return os.path.join(root, dir, tile_id, patch_id)

        folders = [
            {
                "s1": construct_folder_path(self.root, dir_s1, row['s1_name'] ,3),
                "s2": construct_folder_path(self.root, dir_s2, row['patch_id'],2),
                "maps": construct_folder_path(self.root, dir_maps, row['patch_id'],2),
            }
            for _, row in self.metadata.iterrows()
        ] 

        return folders

    def _load_map_paths(self, index: int) -> list[str]:
        """Load paths to band files.

        Args:
            index: index to return

        Returns:
            list of file paths
        """
        folder_maps = self.folders[index]["maps"]
        paths_maps = glob.glob(os.path.join(folder_maps, "*_reference_map.tif"))
        paths_maps = sorted(paths_maps)
        return paths_maps

    def _load_paths(self, index: int, bands) -> list[str]:
        """Load paths to band files.

        Args:
            index: index to return
            bands: either of ['s1', 's2', 'all']

        Returns:
            list of file paths
        """

        if bands == "all":
            folder_s1 = self.folders[index]["s1"]
            folder_s2 = self.folders[index]["s2"]
            paths_s1 = glob.glob(os.path.join(folder_s1, "*.tif"))
            paths_s2 = glob.glob(os.path.join(folder_s2, "*.tif"))
            paths_s1 = sorted(paths_s1)
            paths_s2 = sorted(paths_s2, key=sort_sentinel2_bands)
            paths = paths_s1 + paths_s2
        elif bands == "s1":
            folder = self.folders[index]["s1"]
            paths = glob.glob(os.path.join(folder, "*.tif"))
            paths = sorted(paths)
        else: #s2
            folder = self.folders[index]["s2"]
            paths = glob.glob(os.path.join(folder, "*.tif"))
            paths = sorted(paths, key=sort_sentinel2_bands)

        return paths

    def _load_image(self, index: int, bands:List) -> Tensor:
        """Load a single image.

        Args:
            index: index to return
            bands: either of ['s1', 's2', 'all']
        Returns:
            the raster image or target
        """
        paths = self._load_paths(index, bands)
        images = []
        for path in paths:
            # Bands are of different spatial resolutions
            # Resample to (120, 120)
            with rasterio.open(path) as dataset:
                array = dataset.read(
                    indexes=1,
                    out_shape=self.image_size,
                    out_dtype="int32",
                    resampling=Resampling.bilinear,
                )
                images.append(array)

        arrays: "np.typing.NDArray[np.int_]" = np.stack(images, axis=0)
        return torch.from_numpy(arrays).float()
        

    def _load_map(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the raster image or target
        """
        paths = self._load_map_paths(index)
        map = None
        for path in paths:
            with rasterio.open(path) as dataset:
                map = dataset.read(
                    # indexes=1,
                    # out_shape=self.image_size,
                    out_dtype="int32",
                    # resampling=Resampling.bilinear,
                )

        #TODO: convert output values from 43 to 19 classes if needed
        # if self.num_classes == 19:
        #     map = np.vectorize(self.label_converter.get)(map)
        #     map = np.where(map == None, 0, map)

        return torch.from_numpy(map).float()


    def _load_target(self, index: int) -> Tensor:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target label
        """

        image_labels = self.metadata.iloc[index]['labels']

        # labels -> indices
        indices = [self.class2idx(label) for label in image_labels]

        # Map 43 to 19 class labels: no need, image labels are in 19 class already
        # if self.num_classes == 19:
        #     indices_optional = [self.label_converter.get(idx) for idx in indices]
        #     indices = [idx for idx in indices_optional if idx is not None]

        image_target = torch.zeros(self.num_classes, dtype=torch.long)
        image_target[indices] = 1

        return image_target

    def _onehot_labels_to_names(
        self, label_mask: "np.typing.NDArray[np.bool_]"
    ) -> list[str]:
        """Gets a list of class names given a label mask.

        Args:
            label_mask: a boolean mask corresponding to a set of labels or predictions

        Returns
            a list of class names corresponding to the input mask
        """
        labels = []
        for i, mask in enumerate(label_mask):
            if mask:
                labels.append(self.class_sets[self.num_classes][i])
        return labels

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.2
        """
        if self.bands == "s2":
            fig, ax = plt.subplots(1, 2, figsize=(14, 7))
            image = np.rollaxis(sample["imgs"][[3, 2, 1]].numpy(), 0, 3)
            image = np.clip(image / 2000, 0, 1)
            ax[0].imshow(image)
            ax[1].imshow(sample['map'][0],cmap='viridis')

            
        elif self.bands == "all":
            fig, ax = plt.subplots(1, 3, figsize=(14, 7))
            s2 = np.rollaxis(sample["imgs"][[5, 4, 3]].numpy(), 0, 3)
            s2 = np.clip(s2 / 2000, 0, 1)
            ax[0].imshow(s2)
            ax[0].set_axis_off()
            s1 = sample["imgs"][[0, 1]].numpy()
            #create a third channel with difference of VV and VH
            s1 = np.stack((s1[0], s1[1], s1[0] - s1[1]), axis=0)
            #normalize each s1 band
            s1 = np.clip(s1 / -40, 0, 1)
            s1 = np.rollaxis(s1, 0, 3)
            im1 = ax[1].imshow(s1)
            ax[1].set_axis_off()
            im1.set_clim(-20,0)
            ax[2].imshow(sample['map'][0],cmap='viridis')


        elif self.bands == "s1":
            image = sample["image"][0].numpy()

        label_mask = sample["label"].numpy().astype(np.bool_)
        labels = self._onehot_labels_to_names(label_mask)

        showing_predictions = "predictions" in sample
        if showing_predictions:
            prediction_mask = sample["predictions"].numpy().astype(np.bool_)
            predictions = self._onehot_labels_to_names(prediction_mask)

        plt.axis("off")
        if show_titles:
            title = f"Labels: {', '.join(labels)}"
            if showing_predictions:
                title += f"\nPredictions: {', '.join(predictions)}"
            plt.title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig



class BigEarthNetv2Wrapper(BigEarthNetv2):
    #Mean and STDEV copied from https://github.com/lhackel-tub/ConfigILM/blob/main/configilm/extra/BENv2_utils.py ["120_bilinear"] and band order adjusted to match torchgeo
    MEAN: dict = { "s2" : torch.tensor([ 360.64678955078125, 438.3720703125, 614.0556640625, 588.4096069335938, 942.7476806640625,
                                        1769.8486328125, 2049.475830078125, 2193.2919921875, 2235.48681640625, 2241.10595703125, 1568.2115478515625, 997.715087890625]) ,
                "s1": torch.tensor([-19.352558135986328, -12.643863677978516,])}
    STD: dict = { "s2" : torch.tensor([563.1734008789062, 607.02685546875,603.2968139648438,684.56884765625,727.5784301757812,
                                        1087.4288330078125,1261.4302978515625,1369.3717041015625,1342.490478515625,1294.35546875, 1063.9197998046875,806.8846435546875]),
                "s1": torch.tensor([5.590505599975586, 5.133493900299072])}


    def __init__(self,
        root: str = "data",
        split: str = "train",
        bands: str = "all",
        num_classes: int = 19,
        transform=None,
        #-------Panopticon options-------#
        timestamp_format:str = 'days_since_2002-01-01',
        normalize: bool = True, #whether to normalize the images
        multimodal:bool = True, #whether to send data as separate modalities (True) or together (False)
        full_spectra:bool = False #whether to return both mu and sigma for each band
        #--------------------------------#
        
    ) -> None:
        """Initialize a new BigEarthNet dataset wrapper."""
        super().__init__(root, split, bands, num_classes, transform, download=False, checksum=False)

        if 'days_since_' in timestamp_format:
            # extract the number after 'days since'
            self.ref_date = timestamp_format.split('days_since_')[1]
            self.ref_date = datetime.strptime(self.ref_date, '%Y-%m-%d')
            logger.info(f'Using reference date: {self.ref_date}')
        else:
            logger.info(f'Using timestamp format: {timestamp_format}')
        
        self.normalize = normalize
        self.multimodal = multimodal

        if self.multimodal:
            assert self.bands == 'all', "Multimodal option is only available for all bands"

        if self.bands == 'all' and not self.multimodal:
            self.ds_name = 'ben-s1s2'
            self.chn_ids = torch.tensor(extract_wavemus(load_ds_cfg(self.ds_name), return_sigmas=full_spectra), dtype=torch.long)
        else: #cases where output is a single sensor -> s1 or s2 or all with multimodal
            self.sensor_name_mapping: dict = {'s1' : 'ben-s1',  's2': 'ben-s2'}
            self.chn_ids = {k: extract_wavemus(load_ds_cfg(k), return_sigmas=full_spectra) for k in self.sensor_name_mapping.values() }

        if self.normalize:
            self.channelwise_transforms = self._build_ch_transforms()

    #Override the __get_item__
    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """

        if self.multimodal:
            imgs = [self._load_image(index, b) for b in self.sensor_name_mapping.keys()]
            if self.normalize:
                imgs = [self.channelwise_transforms[sensor](imgs[i]) for i, sensor in enumerate(self.sensor_name_mapping.keys())]
            chn_ids = [self.chn_ids[sensor] for sensor in self.sensor_name_mapping.values()]

        else:
            imgs = self._load_image(index, self.bands)
            if self.normalize:
                imgs = self.channelwise_transforms[self.bands](imgs)
            if self.bands == 'all':
                chn_ids = self.chn_ids
            else:
                chn_ids = self.chn_ids[self.sensor_name_mapping[self.bands]]
            
        # times = self._load_time(index)
        label = self._load_target(index)
        img_map = self._load_map(index) # segmentation map
        sample: dict[str, Tensor] = {"imgs": imgs, 
                                    "chn_ids" : chn_ids,}
                                    # "time" : times,

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample, label

    def _format_timestamp(self, ts):
        if self.ref_date is None:
            #convert from object to numpy datetime64 to python datetime
            ts = ts.astype('M8[D]').astype('O')
            return torch.tensor([ts.year, ts.month, ts.day], dtype=torch.int16)

        else:
            ts = datetime.strptime(ts, '%Y%m%d')
            #convert ref_date from string to python datetime
            delta = (ts - self.ref_date).days
            return torch.tensor(delta, dtype=torch.long)

    def _load_time(self, index: int) -> Tensor:
        """Load the time for a single image.

        Args:
            index: index to return

        Returns:
            the time the S2 image was taken
        """
        timestamp = self.metadata.iloc[index]['s2v1_name']
        print(timestamp, timestamp.split('_')[2].split('T')[0], self.ref_date, type(timestamp))
        return self._format_timestamp(timestamp.split('_')[2].split('T')[0])

    def _load_gsd(self, index: int) -> Tensor:
        """Load the time for a single image.

        Args:
            index: index to return

        Returns:
            the time the S2 image was taken
        """
        gsd = 10 #always
        return torch.tensor(gsd, dtype=torch.float32)

    def _build_ch_transforms(self):
        channelwise_transforms = {}
        for sensor in self.MEAN.keys():
            channelwise_transforms[sensor] = transforms.Normalize(self.MEAN[sensor], self.STD[sensor])

        #create another key for "all" which is the mean and std of all bands
        # all_mean = torch.cat([self.MEAN[sensor] for sensor in self.MEAN.keys()])
        # all_std = torch.cat([self.STD[sensor] for sensor in self.STD.keys()])
        all_mean = torch.cat([self.MEAN[sensor] for sensor in ['s1','s2']])
        all_std = torch.cat([self.STD[sensor] for sensor in ['s1','s2']])
        channelwise_transforms["all"] = transforms.Normalize(all_mean, all_std)

        return channelwise_transforms

    def get_targets(self):
        return np.arange(self.num_classes)



########## Webdataset

def build_tars_benv2_webdataset(
        benv2_ds,
        nshards = math.inf,
        nfiles_per_shard = 10, # 1480
        out_base = None,
        dsname = 'dataset',
        overwrite = False,
        rm_untarred_dirs = False):
    
    os.makedirs(out_base, exist_ok=True)
    metadata_file = os.path.join(out_base, f'{dsname}.yaml')
    if os.path.exists(metadata_file):
        if overwrite:
            logger.info('Overwriting existing dataset.')
        else:
            raise RuntimeError(
                f'{metadata_file} already exists. Set Overwrite=True to continue regardless.')

    # save metainfo
    handle = benv2_ds
    if isinstance(benv2_ds, Subset):
        handle = benv2_ds.dataset
    sensor = handle.bands

    if sensor == 'all':
        sensor = 's1s2'
        mean = torch.cat([handle.MEAN[sensor] for sensor in ['s1','s2']])
        std = torch.cat([handle.STD[sensor] for sensor in ['s1','s2']])
    else:
        mean = handle.MEAN[sensor]
        std = handle.STD[sensor]
    torch.save(dict(mean=mean, std=std), os.path.join(out_base, 'dataset.pt'))

    metadata = dict(
        sensor = sensor,
        num_classes = len(handle.get_targets()))

    with open(metadata_file, 'w+') as f:
        yaml.dump(metadata, f)
    with open(metadata_file, 'a') as f:
        f.write('nsamples_per_shard:\n')

    # build tars

    i = 0
    ishard = 0
    pbar = tqdm(total=min(nshards * nfiles_per_shard, len(benv2_ds)))

    while ishard < nshards and i < len(benv2_ds):

        shard_name = f'{dsname}-{ishard:06d}'
        shard_base = os.path.join(out_base, shard_name)
        os.makedirs(shard_base, exist_ok=True)

        # create files
        i_per_shard = 0
        while i_per_shard < nfiles_per_shard and i < len(benv2_ds):
            idx = i

            # load
            sample, label = benv2_ds[idx]
            img = sample['imgs']
            if sensor == 's1':
                name = handle.metadata.loc[idx, 's1_name']
            elif sensor == 's2':
                name = handle.metadata.loc[idx, 's2v1_name']
            else:
                s1name = handle.metadata.loc[idx, 's1_name']
                s2name = handle.metadata.loc[idx, 's2v1_name']
                name = f'{s1name}_{s2name}'

            #  store image
            imgfile = os.path.join(shard_base, f'{name}.img.npy')
            with open(imgfile, 'wb') as f:
                np.save(f, img.numpy())

            # store labels
            labelfile = os.path.join(shard_base, f'{name}.label.npy')
            with open(labelfile, 'wb') as f:
                np.save(f, label.numpy().astype(bool))

            i += 1
            i_per_shard += 1   
            pbar.update(1) 

        # create tar
        os.system(f'tar --sort=name -cf {os.path.join(out_base,shard_name)}.tar -C {out_base} {shard_name}')
        with open(metadata_file, 'a') as f:
            f.write(f'  {shard_name}.tar: {i_per_shard}\n')

        if rm_untarred_dirs:
            os.system(f'rm -rf {shard_base}')

        ishard += 1

    pbar.close()



def create_benv2_wds(url, normalize=True, transform=None, **kwargs):

    # get metainfo

    metadata = _get_metainfo(url)
    num_classes = metadata['num_classes']
    sensor = metadata['sensor']
    chn_ids = extract_wavemus(load_ds_cfg(f'ben-{sensor}'))

    # get mean & std

    base_dir = os.path.dirname(url)
    mean_std_file = [f for f in os.listdir(base_dir) if f.endswith('.pt')]
    assert len(mean_std_file) == 1, f'Expected 1 .pt file, found {len(mean_std_file)}'
    mean_std = torch.load(os.path.join(base_dir, mean_std_file[0]))

    mu, sigma = mean_std['mean'], mean_std['std']
    norm_trf = transforms.Normalize(mu, sigma)

    # build wds

    def make_sample(sample):
        imgs = torch.from_numpy(sample['img.npy'])
        if normalize:
            imgs = norm_trf(imgs)
        x_dict = dict(imgs=imgs, chn_ids=chn_ids)
        x_dict = transform(x_dict)

        label = torch.from_numpy(sample['label.npy']).int()
        
        return x_dict, label

    trainset = make_wds(url, make_sample, **kwargs)
    trainset.num_classes = num_classes
    trainset.is_webdataset = True

    return trainset



if __name__ == '__main__':
    """ build benv2 webdataset """

    root = None

    for split in ['train', 'validation', 'test']:
        for bands in ['all','s2','s1']:
            print(f'{split} - {bands}')

            ds = BigEarthNetv2Wrapper(
                split = split,
                bands = bands,
                root = root,
                multimodal = False,
                normalize = False,  # normalization done in make_sample in webdataset!
                )

            indices = list(range(len(ds)))
            random.seed(42)
            random.shuffle(indices)
            ds = Subset(ds, indices)

            nfiles_per_shard = math.ceil(len(ds)/100)
            print(f'{nfiles_per_shard} files per shard')

            build_tars_benv2_webdataset(
                ds,
                # nshards=2,
                nfiles_per_shard = nfiles_per_shard,
                out_base=f'{root}/wds/{split}-{bands}',
                dsname='dataset',
                overwrite=False,
                rm_untarred_dirs=True)