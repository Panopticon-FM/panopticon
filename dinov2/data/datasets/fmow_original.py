from dinov2.utils.data import load_ds_cfg, extract_wavemus
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import torch
from torch.functional import F
from torchvision import transforms
from PIL import Image
import logging
from tifffile import imread
import datetime
import traceback

logger = logging.getLogger("dinov2")

class FmowDatasetOriginal(Dataset):

    # Define class-level constants for mean and standard deviation of each band
    MEAN: dict = { "fmow_s2" : torch.tensor([1569.1970, 1383.9951, 1338.4231, 1408.1113, 1537.2856,
                                 1926.5183, 2136.5127, 2033.4019, 2262.4558, 674.1297, 
                                 16.7465, 2027.3674, 1501.4686]) ,
            "fmow_wv23": torch.tensor([324.14698805992043, 321.9731490132023, 414.5839148154745, 376.7135672751123, 287.4754539285566, 400.37182819120585, 455.80341042344526, 387.41375747632117]), #over 20% of train, 55152 samples 
            "fmow_qbge": torch.tensor([444.3088455498445, 427.3245457864162, 306.4215948690728, 576.8987799591143]), #over 40% of train, 35123 samples 
            "fmow_rgb":  torch.tensor([104.55944194258717, 106.19051077034885, 100.09524832336331]), # rgb highres over 10% of train (all 4 sensors, 36357 samples)
    }

    STD: dict = { "fmow_s2" : torch.tensor([517.7932, 595.0351, 690.2477, 994.7734, 994.4599, 1079.4558, 1192.3668,
                    1170.9979, 1278.2103, 474.7933, 15.1294, 1362.3807, 1134.5983]),
            "fmow_wv23": torch.tensor([115.78370553994293, 134.64109966204413, 210.5490582067263, 242.05723188930372, 204.35781294894136, 246.31516378243006, 282.0780383349591, 255.94032664657144]),
            "fmow_qbge": torch.tensor([229.0287, 247.0869, 239.2377, 381.1768]),   
            "fmow_rgb":  torch.tensor([68.97789839437421, 66.91165970478865, 69.09505694641828]), # rgb highres over 10% of train (all 4 sensors, 36357 samples)
    }

    sensor_name_mapping: dict = {'WORLDVIEW02' : 'fmow_wv23', 
                                 'WORLDVIEW03_VNIR' : 'fmow_wv23',
                                 'GEOEYE01': 'fmow_qbge',
                                 'QUICKBIRD02': 'fmow_qbge',
                                 's2': 'fmow_s2', 
                                 'fmow_rgb': 'fmow_rgb'
                                }
    RGB_CHANNELS: dict = {
        'fmow_s2': [3, 2, 1],
        'fmow_wv23': [4, 2, 1], 
        'fmow_rgb': [0, 1, 2],  
    }

    CLASS_NAMES = ["airport", "airport_hangar", "airport_terminal", "amusement_park", "aquaculture", "archaeological_site", "barn", "border_checkpoint", "burial_site", "car_dealership", "construction_site", "crop_field", "dam", "debris_or_rubble", "educational_institution", "electric_substation", "factory_or_powerplant", "fire_station", "flooded_road", "fountain", "gas_station", "golf_course", "ground_transportation_station", "helipad", "hospital", "impoverished_settlement", "interchange", "lake_or_pond", "lighthouse", "military_facility", "multi-unit_residential", "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park", "parking_lot_or_garage", "place_of_worship", "police_station", "port", "prison", "race_track", "railway_bridge", "recreational_facility", "road_bridge", "runway", "shipyard", "shopping_mall", "single-unit_residential", "smokestack", "solar_farm", "space_facility", "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth", "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility", "wind_farm", "zoo"]
    #It was found empirically that some wv2 or wv3 images are not 8 channels, so for simplicty we just drop them
    PROBLEMATIC_IDS = { 'train' :[172, 3683, 4279, 4368, 5326, 5687, 6561, 6584, 6594, 7469, 7613, 7631, 7665, 7845, 8223, 8324, 11319, 11427, 12255, 13304, 18180, 19772, 19806, 20164, 20793, 20803, 21315, 21419, 21834, 21866, 22297, 22439, 22927, 23509, 23927, 24163, 25414, 25417, 26003, 26218, 26506, 26887, 27351, 28676, 28806, 29182, 29352, 29537, 32106, 32108, 34798, 35256, 37086, 37202, 38592, 39566, 39588, 40009, 40229, 40254, 41861, 46695, 47180, 50438, 55145, 55514, 56884, 57879, 65546, 66268, 66948, 76512, 76751, 76922, 77810, 83412, 84186, 86885, 86972, 87796, 87998, 88326, 88510, 89445, 89471, 90259, 92009, 92223, 92294, 93323, 94122, 94208, 94598, 94884, 95296, 96003, 96227, 98881, 99112, 99537, 100331, 100991, 101475, 101575, 101735, 101865, 101887, 102162, 102178, 102594, 102804, 104402, 104640, 104683, 104848, 104900, 105202, 105396, 106584, 107272, 107726, 108213, 108324, 108359, 109860, 110119, 110181, 110686, 111120, 111750, 112090, 113025, 113353, 113519, 114633, 115387, 116286, 116382, 116898, 117864, 117927, 118745, 119110, 119218, 119843, 120279, 
                                120435, 120648, 120788, 120981, 121285, 121659, 121854, 123311, 124087, 124363, 124951, 125729, 128115, 129730, 129755, 129791, 131253, 131588, 131640, 132041, 132184, 133437, 134181, 135473, 136125, 136313, 136476, 137200, 137375, 137459, 138229, 139407, 139675, 140131, 140477, 140500, 140945, 141432, 141803, 142053, 142490, 142820, 143242, 143365, 144591, 145366, 149221, 151137, 151660, 151678, 152306, 153496, 156649, 156679, 160737, 163513, 164420, 164570, 164968, 165045, 165408, 166211, 166293, 166340, 166657, 182497, 182662, 183149, 184329, 185095, 185195, 189484, 192172, 192625, 193385, 194336, 196235, 197945, 198011, 198590, 200333, 203800, 203805, 211778, 211878, 212600, 215136, 215494, 215872, 215925, 216563, 218300, 219464, 221222, 222584, 222642, 224149, 225182, 226640, 227253, 229416, 230218, 231262, 231805, 232172, 232233, 232363, 241886, 242255, 243623, 246673, 248193, 249064, 250120, 251552, 251751, 251818, 252316, 252335, 252516, 252537, 252592, 252691, 252943, 253089, 253098, 253323, 254363, 255377, 255970, 255971, 256167, 
                                256281, 256740, 257490, 257630, 258549, 258570, 260906, 262962, 263069, 266104, 266374, 266381, 266516, 267471, 268160, 270528, 271482, 272303, 272752, 272971, 273176, 274066, 274162, 274202, 274972],
                        'val': [1033, 1261, 1825, 2923, 2925, 2957, 2997, 3012, 3060, 3300, 3754, 5695, 8468, 9921, 11259, 13306, 14176, 14537, 14563, 14808, 15055, 15193, 15970, 16059, 16116, 16120, 16807, 17073, 17343, 18956, 19380, 19918, 20051, 20588, 23609, 23615, 27873, 30238, 31685, 32007, 32014, 32027, 32161, 35368, 35944, 36897, 37339, 37819, 38194]
    }

    def __init__(self, split='train', 
                 normalize=True, 
                 root='${RDIR}/datasets',
                 keep_sensors=['WORLDVIEW02', 'WORLDVIEW03_VNIR', 'GEOEYE01', 'QUICKBIRD02'],
                 return_rgb=False,
                 transform=None,
                 full_spectra=False,
                 output_dtype='float32',
                 min_img_size:int = None,
                 max_img_size:int = None,):
        """

        split: str, one of ['train', 'val', 'test']
        normalize: bool, whether to normalize the images
        root: str, root directory of the dataset
        transform: callable, transform to apply to the images
        max_crop: int, each row must have its minimum size >= this (prevents fmow from generating all images with a small size)
        min_crop: int, each row must have all its image size >= this (ensures fmow generating all images above this min threshold)
        keep_sensors: list of str, sensors to keep in the dataset, one of ['wv23', 's2', 'rgb']
        """

        root = os.path.expandvars(root)
        self.transform = transform
        self.return_rgb = return_rgb
        self.num_classes = len(self.CLASS_NAMES)

        assert isinstance(output_dtype, str)
        if output_dtype == 'float16':
            torch_dtype, np_dtype = torch.float16, np.float16
        elif output_dtype == 'float32':
            torch_dtype, np_dtype = torch.float32, np.float32
        elif output_dtype == 'float64':
            torch_dtype, np_dtype = torch.float64, np.float64
        else:
            raise ValueError(f'Unknown output_dtype: {output_dtype}')
        self.torch_dtype, self.np_dtype = torch_dtype, np_dtype
        logger.info(f'output_dtype: {output_dtype}')

        # read file metainfo
        if split in ['train', 'val']: 
            metadata_path = os.path.join(root, f'fmow/metadata_v2/fmow_{split}.parquet')
        elif split is None: #pick train
            metadata_path = os.path.join(root, 'fmow/metadata_v2/fmow_train.parquet')
        elif split == 'test':
            metadata_path = os.path.join(root, 'fmow/metadata_v2/fmow_test_gt.parquet')
        else:
            metadata_path = os.path.join(root, os.path.expandvars(split))

        self.df = pd.read_parquet(metadata_path)
        
        # load dataset metainfo
        ds_names = self.sensor_name_mapping.values()
        self.chn_ids = {k: extract_wavemus(load_ds_cfg(k), full_spectra) for k in ds_names } 

        self.normalize = normalize
        self.root = root
        self.min_img_size = min_img_size
        self.max_img_size = max_img_size
        self.keep_sensors = keep_sensors
        self.split = split
        self._subset_df()

        self.log_stats()
        self.problematic_ids = []

        if self.normalize:
            logger.info('Normalizing images')
            self.channelwise_transforms = self._build_ch_transforms()

    def _subset_df(self):

        if not self.return_rgb:
            self.df = self.df[self.df['ms_sensor_platform_name'].isin(self.keep_sensors)]
        else:
            self.df = self.df[self.df['rgb_sensor_platform_name'].isin(self.keep_sensors)]
            #filter out all rows where rgb_is_corrupt is True
            # self.df = self.df[self.df['rgb_is_corrupt'] == False]

        #remove row in PROBLEMATIC_IDS
        if self.split in self.PROBLEMATIC_IDS:
            problematic_ids = self.PROBLEMATIC_IDS[self.split]
            self.df = self.df[~self.df.index.isin(problematic_ids)]
            logger.info(f'Removed {len(problematic_ids)} problematic images')
            print(f'Removed {len(problematic_ids)} problematic images')
        

    def _build_ch_transforms(self):
        channelwise_transforms = {}
        for sensor in self.MEAN.keys():
            channelwise_transforms[sensor] = transforms.Normalize(self.MEAN[sensor], self.STD[sensor])
        return channelwise_transforms

    def log_stats(self):
        sensor_counts = {sensor: 0 for sensor in self.sensor_name_mapping.keys()}
        for sensor in self.sensor_name_mapping.keys():
            sensor_counts[sensor] = self.df['ms_sensor_platform_name'].apply(lambda x: sensor in x).sum()
        logger.info(f'Dataset size: {self.__len__()}, sensor counts: {sensor_counts}')


    def _load_img(self, path) -> torch.Tensor:
        path = os.path.join(self.root, path)
        return torch.from_numpy(imread(path).astype(self.np_dtype)).permute(2,0,1)

    def _get_label(self, id:str):
        #given a string id, return the index of the label in the CLASS_NAMES list
        try:
            return self.CLASS_NAMES.index(id)
        except ValueError:
            logger.warning(f'Label not found in CLASS_NAMES: {id}')
            return None


    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.return_rgb:
            key = 'rgb_path'
            ds_name, sensor = 'fmow_rgb', 'fmow_rgb'
        else:
            key = 'ms_path'
            ds_name, sensor = self.sensor_name_mapping[row['ms_sensor_platform_name']], self.sensor_name_mapping[row['ms_sensor_platform_name']]
        try: 
            img = self._load_img(row[key])
        except Exception as e:
            faulty_path = os.path.join(self.root, row[key])
            full_traceback = traceback.format_exc()

            logger.info(f'Error loading image: {faulty_path}')
            logger.info(full_traceback)
            # if self.faulty_imgs_file is not None:
            #     with open(self.faulty_imgs_file, 'a') as f:
            #         f.write('\n\n')
            #         f.write('time: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')
            #         f.write('file: ' + faulty_path + '\n')
            #         f.write(full_traceback)
            self.problematic_ids.append(idx)
            return None, None
        chn_id = self.chn_ids[ds_name]

        label_id = '_'.join(row['id'].split('_')[:-2])
        label = self._get_label(label_id)

        if (img.shape[0] == 8 and sensor != 'fmow_wv23') or (img.shape[0] == 4 and sensor != 'fmow_qbge'):
            print(img.shape, sensor, idx)
            self.problematic_ids.append(idx)
    
        if self.normalize:
            img = self.channelwise_transforms[sensor](img)


        x_dict = dict(imgs=img, chn_ids=chn_id)

        if self.transform is not None:
            x_dict = self.transform(x_dict)

        return x_dict, label
