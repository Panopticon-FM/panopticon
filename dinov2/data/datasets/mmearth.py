"""MMEarth Dataset."""

from dinov2.utils.data import load_ds_cfg, extract_wavemus
import ujson as json
from argparse import Namespace
from collections import OrderedDict
from pathlib import Path
from typing import Union
import pandas as pd

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import os


#Rewritten to use a simple parquet file for all the metadata
class MMEarth(Dataset):

    NO_DATA_VAL = {
        "sentinel2": 0,
        "sentinel2_cloudmask": 65535,
        "sentinel2_cloudprod": 65535,
        "sentinel2_scl": 255,
        "sentinel1": float("-inf"),
        "aster": float("-inf"),
        "canopy_height_eth": 255,
        "dynamic_world": 0,
        "esa_worldcover": 255,
        "lat": float("-inf"),
        "lon": float("-inf"),
        "month": float("-inf"),
        "era5": float("inf"),
        "biome": 255,
        "eco_region": 65535,
    }


    # an example of all the modalities. DO NOT CHANGE THIS, ALWAYS CHANGE THE INP and OUT MODALITIES ABOVE
    MODALITIES_FULL = {
        "sentinel2": [
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8A",
            "B8",
            "B9",
            "B10",
            "B11",
            "B12",
        ],
        # These fields are not used in the current version of the dataset
        "sentinel2_cloudmask": ["QA60"],
        "sentinel2_cloudprod": ["MSK_CLDPRB"],
        "sentinel2_scl": ["SCL"],
        "sentinel1": [
            "asc_VV",
            "asc_VH",
            "asc_HH",
            "asc_HV",
            "desc_VV",
            "desc_VH",
            "desc_HH",
            "desc_HV",
        ],
        "aster": ["elevation", "slope"],
        "era5": [
            "prev_month_avg_temp",
            "prev_month_min_temp",
            "prev_month_max_temp",
            "prev_month_total_precip",
            "curr_month_avg_temp",
            "curr_month_min_temp",
            "curr_month_max_temp",
            "curr_month_total_precip",
            "year_avg_temp",
            "year_min_temp",
            "year_max_temp",
            "year_total_precip",
        ],
        "dynamic_world": ["landcover"],
        "canopy_height_eth": ["height", "std"],
        "lat": ["sin", "cos"],
        "lon": ["sin", "cos"],
        "biome": ["biome"],
        "eco_region": ["eco_region"],
        "month": ["sin_month", "cos_month"],
        "esa_worldcover": ["map"],
    }

    MODALITY_MINIMAL_SET1 = {
        "sentinel2": [
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8A",
            "B8",
            "B9",
            "B10",
            "B11",
            "B12",
        ],
        "sentinel1": [
            "asc_VV",
            "asc_VH",
            "asc_HH",
            "asc_HV",
            "desc_VV",
            "desc_VH",
            "desc_HH",
            "desc_HV",
        ],
        "month": ["sin_month", "cos_month"],
    }

    MODALITY_S1 = {k:v for k,v in MODALITY_MINIMAL_SET1.items() if k=='sentinel1'}
    MODALITY_S2 = {k:v for k,v in MODALITY_MINIMAL_SET1.items() if k=='sentinel2'}

    MODALITY_TASK = {
        # map regression
        "sentinel2": "regression_map",
        "sentinel1": "regression_map",
        "aster": "regression_map",
        "canopy_height_eth": "regression_map",
        # pixel regression
        "lat": "regression",
        "lon": "regression",
        "month": "regression",
        "era5": "regression",
        # semantic segmentation
        "esa_worldcover": "segmentation",
        "dynamic_world": "segmentation",
        # pixel classification
        "biome": "classification",
        "eco_region": "classification",
    }

    PIXEL_WISE_MODALITIES = [
        "sentinel2",
        "sentinel1",
        "aster",
        "canopy_height_eth",
        "esa_worldcover",
        "dynamic_world",
    ]
    
    def __init__(self, base_path, modalities=MODALITIES_FULL, split="train"):

        assert split in ["train", "eval", "all"], "split must be one of ['train', 'eval', 'all']"


        variant = base_path.split("/")[-1] # "data_1M_v001"
        self.data_path = base_path + f'/{variant}.h5' #hdf5 file
        self.meta_path = base_path + f'/{variant}.parquet' #parquet file
        self.band_stats_path = base_path + f'/{variant}_band_stats.json'

        print(variant, self.data_path)
        self.modalities = modalities  # modalities used for training
        self.modalities_full = (
            self.MODALITIES_FULL
        )  # all modalities present in the datasets. This is used to keep track of the indices of the modalities in the dataset.
        
        self._open_parquet(self.meta_path)

        if split != "all":
            self.meta_full = self.meta_full[self.meta_full["split"] == split].reset_index(drop=True)
            # raise NotImplementedError("Splitting is not available in this dataset")
        self._filter_metadata()
        self.norm_stats = json.load(open(Path(self.band_stats_path), "r"))

    def _open_parquet(self, path):
        try:
            self.meta_full = pd.read_parquet(path)
        except:
            raise FileNotFoundError(f"Metadata file not found at {path}. Please generate the file and then continue")

    def _filter_metadata(self):
        """
        Ensures that the metadata only contains the rows that have atleast ascending or descending data.
        """
        len1 = len(self.meta_full)
        def contains_sentinel1(sensor_list):
            return any('sentinel1_asc' in sensor or 'sentinel1_desc' in sensor for sensor in sensor_list)

        # Apply the function to create a boolean mask
        mask = self.meta_full['sensor'].apply(contains_sentinel1)

        # Filter the DataFrame with the mask
        self.meta_full = self.meta_full[mask]
        self.meta_full.reset_index(drop=True, inplace=True)
        len2 = len(self.meta_full)
        print(f"Removed {len1 - len2} rows that did not have sentinel1 data")
        #check if there are any nans in hf5_idx
        if self.meta_full['hf5_idx'].isnull().sum() > 0:
            raise ValueError("hf5_idx contains nans")

        self.meta_full['hf5_idx'] = self.meta_full['hf5_idx'].astype(int)

    def _generate_metadata_file(self, base_path):
        
        variant = base_path.split("/")[-1] # "data_1M_v001"
        # metadata files
        with open(f"{base_path}/{variant}/{variant}_tile_info.json", "r") as file:
            tile_data = json.load(file)
        with open(f"{base_path}/{variant}/{variant}_splits.json", "r") as file:
            split_data = json.load(file)

        # read the hf5 file
        with h5py.File(f"{base_path}/{variant}/{variant}.h5", "r") as file:
            hf5_ids = [entry[0].decode() for entry in file["metadata"]]

        df_mm = pd.DataFrame.from_dict(tile_data, orient='index')
        df_mm.reset_index(inplace=True)
        df_mm.rename(columns={'index': 'id'}, inplace=True)

        split_df = pd.DataFrame([(key, split) for split, keys in split_data.items() for key in keys], columns=['hf5_idx', 'split'])
        split_df.sort_values(by='hf5_idx', inplace=True)
        split_df["id"] = hf5_ids

        df_mm = df_mm.merge(split_df, on='id', how='left')

        df_mm['sensor'] = df_mm['BANDS'].apply(lambda x: [k for k, v in x.items() if v is not None])

        # MM-Earth only specifies training samples, so possible "val" and "test" samples are marked as "eval"
        df_mm['split'] = df_mm['split'].fillna('eval')

        df_mm.columns = df_mm.columns.str.lower()

        # "id" is some unique id that provides a link between the metadata.json file and the hf5 file["metadata"]
        # to get the actual modality sensor data from the hf5 file, we need to use the "hf5_idx" column
        # an accessed sample from the hf5 file will have a "tile_id" that matches the "id" column in the metadata.json file
        # i.e assert f["metadata"][hf5_idx][0].decode('utf-8') in self.meta_tile_info
        df_mm.to_parquet(f"{base_path}/{variant}.parquet")

    def _open_hdf5(self, path):
        self.data_full = h5py.File(path, "r")

    def __len__(self):
        return len(self.meta_full)

    def __getitem__(self, idx):

        # this is to ensure that multiple workers do not open the same file multiple times.
        if not hasattr(self, "data_full"):
            self._open_hdf5(self.data_path)

        # based on what bands and what modalities we need for training, we return the data[idx].)
        return_dict = OrderedDict()

        row = self.meta_full.iloc[idx]
        hf5_idx = row["hf5_idx"]

        # print(f'idx {idx}, hf5_idx {hf5_idx}')
        # print(f'available bands: {row["bands"]}')

        l2a = row['s2_type'] == "l2a"

        for modality in self.modalities.keys():
            # get the indices based on how it is in modalities_full
            if self.modalities[modality] == "all":
                modality_idx = [i for i in range(len(self.modalities_full[modality]))]
            else:
                modality_idx = [
                    self.modalities_full[modality].index(m)
                    for m in self.modalities[modality]
                ]

            if modality in ["biome", "eco_region"]:
                # for these modalities the array is already one hot encoded. hence modality_idx is not needed.
                data = self.data_full[modality][hf5_idx, ...]

            else:
                # get the data
                data = self.data_full[modality][hf5_idx, modality_idx, ...]

            data = np.array(data)

            # inside the band_stats, the name for sentinel2 is sentinel2_l1c or sentinel2_l2a
            if modality == "sentinel2":
                modality_ = "sentinel2_l2a" if l2a else "sentinel2_l1c"
            else:
                modality_ = modality

            if modality not in [
                "biome",
                "eco_region",
                "dynamic_world",
                "esa_worldcover",
            ]:
                means = np.array(self.norm_stats[modality_]["mean"])[modality_idx]
                stds = np.array(self.norm_stats[modality_]["std"])[modality_idx]
                if modality in ["era5", "lat", "lon", "month"]:
                    # single value mean and std
                    data = (data - means) / stds
                else:
                    # single value mean and std for each band
                    data = (data - means[:, None, None]) / stds[:, None, None]

            if modality == "dynamic_world":
                # the labels of dynamic world are 0, 1, 2, 3, 4, 5, 6, 7, 8, 9. We convert them to 0, 1, 2, 3, 4, 5, 6, 7, 8, nan respectively.
                # originally when downloading the no data values are 0. hence we remap them to nan.
                data = np.where(data == self.NO_DATA_VAL[modality], np.nan, data)
                old_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan]
                new_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, np.nan]
                for old, new in zip(old_values, new_values):
                    data = np.where(data == old, new, data)
                # for any value greater than 8, we map them to nan
                data = np.where(data > 8, np.nan, data)

            if modality == "esa_worldcover":
                # the labels of esa worldcover are 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 255. We convert them to 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 255 respectively.
                old_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 255]
                new_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255]
                for old, new in zip(old_values, new_values):
                    data = np.where(data == old, new, data)
                # for any value greater than 10, we map them to nan
                data = np.where(data > 10, np.nan, data)

            # converting the nodata values to nan to keep everything consistent
            data = (
                np.where(data == self.NO_DATA_VAL[modality], np.nan, data)
                if modality != "dynamic_world"
                else data
            )

            if self.MODALITY_TASK[modality] in ["classification", "segmentation"]:
                # remap nan values to -1
                data = np.where(np.isnan(data), -1, data)
                data = data.astype(np.dtype("int64"))
            else:
                data = data.astype(np.dtype("float32"))

            return_dict[modality] = data

        # we also return the id, to differentiate between sentinel2_l1c and sentinel2_l2a, since this is given in the tile_info json file.
        # return_dict["id"] = name
        
        return return_dict
    

class MMEarthWrapper(MMEarth):

    def __init__(self, 
                 base_path, 
                 modalities=MMEarth.MODALITIES_FULL, 
                 split='all', 
                 transform=None,
                 full_spectra=False,
                 num_sens=2,
                 remove_nans=True):
        
        if isinstance(modalities, str):
            modalities = MMEarth.__dict__[modalities]
        super().__init__(base_path, modalities, split=split)

        #map from internal sensor names to dataset yaml names
        self.sensor_name_mapping: dict = {'sentinel1' : 'mmearth_s1',  'sentinel2': 'mmearth_s2'}
        self.chn_ids = {k: extract_wavemus(load_ds_cfg(k), full_spectra) for k in self.sensor_name_mapping.values() }
        self.transform = transform
        self.remove_nans = remove_nans
        self.num_sens = num_sens

    #The following is needed because the S1 dataset consists of ascending, descending pairs with upto 4 bands
    # Not all of them are always available. In fact usually just VV, VH for ascending and descending tend to be available
    # Sometimes HH and/or HV will also show up
    # This function will remove any pairs that have nans in them, and remove the corresponding channel ids as well

    
    def check_each_channel_for_nans(self, img, id):
        # Create a boolean mask for channels without NaNs
        mask = ~np.isnan(img).any(axis=(1, 2))

        # Use the mask to filter out channels with NaNs
        filtered_s1 = img[mask]
        filtered_id = id[mask]
        return filtered_s1, filtered_id
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        assert self.modalities == self.MODALITY_MINIMAL_SET1, 'only this is support right now'
        data = {k: v for k, v in data.items() if k in ['sentinel1', 'sentinel2']}

        assert self.num_sens <= len(data), 'Requested more sensors than available'
        keys = np.random.choice(list(data.keys()), self.num_sens, replace=False)
        data = {k: data[k] for k in keys}

        imgs = []
        chn_ids = []

        for k, v in data.items(): #'sentinel1', 'sentinel2'
            if k != 'sentinel1' and k != 'sentinel2': #month
                continue

            modality_idx = [
                    self.modalities_full[k].index(m)
                    for m in self.modalities[k]
                ]

            chn_id = self.chn_ids[self.sensor_name_mapping[k]]
            #also subset according to the channels that are requested
            chn_id = chn_id[modality_idx]
            if self.remove_nans:
                img, chn_id = self.check_each_channel_for_nans(v, chn_id)
            else:
                img = v
            imgs.append(torch.from_numpy(img).float())
            chn_ids.append(chn_id)

        out = [dict(
            imgs = [imgs[i]],
            chn_ids = [chn_ids[i]],
        ) for i in range(len(imgs))]

        if self.transform:
            out = self.transform(out)
            
        return out