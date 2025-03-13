from dinov2.utils.data import load_ds_cfg, extract_wavemus
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import torch
from torch.functional import F
from torchvision import transforms
import logging
from tifffile import imread
import datetime
import traceback
from pathlib import Path

logger = logging.getLogger("dinov2")




class ClipMinMaxNormalize(object):
    def __init__(self, min_val=0.0, max_val=10000.):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor):
        clipped = torch.clamp(tensor, self.min_val, self.max_val)
        return (clipped - self.min_val) / (self.max_val - self.min_val)


class SpectralEarthDataset(Dataset):

    #Mean and Stdev values calculated from 10% random sample of the dataset
    ENMAP_MEAN = [0.0803, 0.0789, 0.0808, 0.0839, 0.0857, 0.0865, 0.0884, 0.0902, 0.0910,
        0.0930, 0.0940, 0.0946, 0.0956, 0.0975, 0.0990, 0.0994, 0.1005, 0.1025,
        0.1033, 0.1056, 0.1091, 0.1110, 0.1132, 0.1163, 0.1193, 0.1225, 0.1246,
        0.1268, 0.1294, 0.1316, 0.1336, 0.1349, 0.1355, 0.1380, 0.1393, 0.1408,
        0.1428, 0.1445, 0.1457, 0.1464, 0.1483, 0.1489, 0.1496, 0.1514, 0.1523,
        0.1521, 0.1529, 0.1538, 0.1545, 0.1556, 0.1593, 0.1639, 0.1733, 0.1826,
        0.1936, 0.2038, 0.2168, 0.2269, 0.2356, 0.2399, 0.2483, 0.2352, 0.2501,
        0.2537, 0.2551, 0.2571, 0.2584, 0.2618, 0.2618, 0.2648, 0.2642, 0.2662,
        0.2673, 0.2684, 0.2707, 0.2712, 0.2731, 0.2714, 0.2738, 0.2819, 0.2721,
        0.2760, 0.2711, 0.2606, 0.2862, 0.2802, 0.2692, 0.2826, 0.2919, 0.2820,
        0.2828, 0.2667, 0.2585, 0.2813, 0.2641, 0.2854, 0.2646, 0.2872, 0.2655,
        0.2823, 0.2729, 0.2907, 0.2932, 0.2962, 0.2994, 0.3021, 0.3057, 0.3096,
        0.3110, 0.3127, 0.3128, 0.3019, 0.3218, 0.3122, 0.2918, 0.2943, 0.2947,
        0.2965, 0.2962, 0.2966, 0.3000, 0.3040, 0.3068, 0.3097, 0.3156, 0.3147,
        0.2305, 0.2342, 0.2381, 0.2417, 0.2455, 0.2491, 0.2524, 0.2565, 0.2599,
        0.2621, 0.2638, 0.2659, 0.2664, 0.2663, 0.2658, 0.2645, 0.2630, 0.2614,
        0.2594, 0.1784, 0.1880, 0.1758, 0.1638, 0.1680, 0.1891, 0.1943, 0.1967,
        0.1868, 0.1893, 0.1895, 0.1959, 0.1934, 0.1937, 0.1950, 0.1965, 0.1960,
        0.1980, 0.1975, 0.1975, 0.1970, 0.1972, 0.1968, 0.1972, 0.1965, 0.1949,
        0.1947, 0.1976, 0.1983, 0.1979, 0.1934, 0.1906, 0.1864, 0.1839, 0.1804,
        0.1796, 0.1744, 0.1722, 0.1692, 0.1663, 0.1658, 0.1654, 0.1605, 0.1602,
        0.1606, 0.1641, 0.1613, 0.1634, 0.1599, 0.1621, 0.1583, 0.1591, 0.1503,
        0.1557, 0.1496, 0.1530, 0.1355]


    ENMAP_STD = [0.1522, 0.1513, 0.1514, 0.1525, 0.1525, 0.1518, 0.1520, 0.1518, 0.1513,
        0.1515, 0.1513, 0.1510, 0.1510, 0.1514, 0.1518, 0.1515, 0.1514, 0.1516,
        0.1512, 0.1514, 0.1518, 0.1513, 0.1506, 0.1503, 0.1503, 0.1507, 0.1506,
        0.1506, 0.1509, 0.1514, 0.1522, 0.1528, 0.1531, 0.1543, 0.1550, 0.1555,
        0.1561, 0.1566, 0.1571, 0.1574, 0.1583, 0.1583, 0.1578, 0.1587, 0.1596,
        0.1597, 0.1602, 0.1607, 0.1610, 0.1608, 0.1618, 0.1588, 0.1559, 0.1521,
        0.1493, 0.1464, 0.1453, 0.1446, 0.1450, 0.1446, 0.1471, 0.1430, 0.1465,
        0.1460, 0.1455, 0.1451, 0.1448, 0.1454, 0.1452, 0.1460, 0.1453, 0.1460,
        0.1458, 0.1458, 0.1457, 0.1441, 0.1441, 0.1422, 0.1416, 0.1387, 0.1395,
        0.1371, 0.1377, 0.1332, 0.1391, 0.1398, 0.1303, 0.1469, 0.1362, 0.1454,
        0.1328, 0.1388, 0.1348, 0.1329, 0.1339, 0.1344, 0.1336, 0.1340, 0.1330,
        0.1331, 0.1331, 0.1330, 0.1331, 0.1335, 0.1344, 0.1354, 0.1371, 0.1385,
        0.1389, 0.1388, 0.1375, 0.1309, 0.1374, 0.1344, 0.1280, 0.1304, 0.1307,
        0.1312, 0.1310, 0.1306, 0.1314, 0.1325, 0.1333, 0.1337, 0.1348, 0.1353,
        0.1442, 0.1441, 0.1441, 0.1439, 0.1439, 0.1438, 0.1437, 0.1442, 0.1444,
        0.1443, 0.1442, 0.1445, 0.1442, 0.1440, 0.1436, 0.1431, 0.1429, 0.1427,
        0.1421, 0.1394, 0.1441, 0.1350, 0.1214, 0.1272, 0.1391, 0.1446, 0.1454,
        0.1393, 0.1398, 0.1404, 0.1435, 0.1419, 0.1409, 0.1417, 0.1410, 0.1406,
        0.1400, 0.1390, 0.1368, 0.1353, 0.1332, 0.1321, 0.1304, 0.1289, 0.1254,
        0.1243, 0.1257, 0.1274, 0.1271, 0.1259, 0.1248, 0.1244, 0.1235, 0.1233,
        0.1226, 0.1204, 0.1187, 0.1191, 0.1166, 0.1166, 0.1160, 0.1140, 0.1126,
        0.1147, 0.1169, 0.1161, 0.1161, 0.1156, 0.1164, 0.1167, 0.1149, 0.1120,
        0.1150, 0.1157, 0.1174, 0.1121]

    #converts from hierarchical to serial order
    # The following classes are removed ion 19-class version and hence mapped to 43 (remove)
    # 122-142: Road and rail networks and associated land , Port areas Airports ,Mineral extraction sites ,Dump sites ,Construction sites ,Green urban areas ,Sport and leisure facilities s versio:
    # 332, 334: Bare rock, Burnt area
    # 423 : Intertidal flats
    CORINE_CLASSES_43 = {111: 0, 112: 1, 121: 2, 122: 43, 123: 43, 124: 43, 131: 43, 132: 43, 133: 43, 141: 43, 142: 43, 
                  211: 11, 212: 12, 213: 13, 221: 14, 222: 15, 223: 16, 231: 17, 241: 18, 242:19, 243: 20, 244: 21, 
                  311: 22, 312: 23, 313: 24, 321: 25, 322: 26, 323: 27, 324: 28, 331: 29, 332: 43, 333: 31, 334: 43, 335:43,
                  411: 33, 412: 34, 421: 35, 422: 36, 423: 43, 
                  511: 38, 512: 39, 521: 40, 522: 41, 523: 42,
                  999: 43} #999 = NODATA

    label_converter = { #from BEN in torchgeo
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
        43: None,
    }

    RGB_CHANNELS: dict = {
        'spectral_earth': [43, 28, 10],
    }

    def __init__(self, 
                 split=None,  #can be one of ['train', 'val', 'test', None]: None is reserved for pretrain only
                 normalize=True, 
                 root='${RDIR}/datasets',
                 task_dir='enmap', # can be one of ['enmap', 'cdl', 'corine', 'nlcd']
                 return_rgb=False,
                 return_multilabel=False,
                 transform=None,
                 faulty_imgs_file=None,
                 full_spectra=False,
                 build_dataset=False,
                 return_chns_by_id=None):

        """
        params:
        - split: str, can be one of ['train', 'val', 'test', None]: None is reserved for pretrain only
        - normalize: bool, whether to normalize the images
        - root: str, root directory for the dataset
        - task_dir: str, name of the task directory, can be one of ['enmap', 'cdl', 'corine', 'nlcd']
            - enmap: pretrain
            - cdl: crop type classification
            - corine: land cover classification
            - nlcd: land cover classification
        - return_rgb: bool, whether to return only RGB channels
        - transform: callable, transform to apply to the dataset
        - faulty_imgs_file: str, path to a file with faulty images
        - full_spectra: bool, whether to return full spectra
        - build_dataset: bool, whether to build the dataset

        """

        root = os.path.expandvars(root)
        self.transform = transform
        self.return_rgb = return_rgb
        self.normalize = normalize
        if return_chns_by_id is not None:
            self.return_chns_by_id = torch.tensor(return_chns_by_id)
        else:
            self.return_chns_by_id = None

        assert split in ['train', 'val', 'test', None], f"split must be one of ['train', 'val', 'test', None], got {split}"
        assert task_dir in ['enmap', 'cdl', 'corine', 'nlcd'], f"task_dir must be one of ['enmap', 'cdl', 'corine', 'nlcd'], got {task_dir}"

        split = 'train' if split == None else split

        if task_dir == 'enmap' and split != 'train':
            raise ValueError('Enmap pretrain task only supports train split')

        self.splits_dir = os.path.join(root, f'spectral_earth/splits/{task_dir}')
        self.data_dir = os.path.join(root, f'spectral_earth/enmap')

        if task_dir != 'enmap':
            self.labels_dir = os.path.join(root, f'spectral_earth/spectral_earth_downstream_datasets/enmap_{task_dir}/{task_dir}')
            # print(f'Labels dir: {self.labels_dir}')
            logger.info(f'SE Labels dir: {self.labels_dir}')
            if task_dir == 'corine':
                self.num_classes = 19
                self.data_dir = os.path.join(root, f'spectral_earth/spectral_earth_downstream_datasets/enmap_{task_dir}/enmap')
            elif task_dir == 'cdl':
                raise NotImplementedError('CDL task not implemented yet')
            elif task_dir == 'nlcd':
                raise NotImplementedError('NLCD task not implemented yet')

        else:
            self.labels_dir = None

        if task_dir == 'enmap' and build_dataset:
            logger.info('Building dataset for spectral earth: ', self.data_dir)
            self.build_dataset(self.splits_dir, self.data_dir)

        # read file metainfo
        if split in ['train', 'val', 'test']:
            metadata_path = os.path.join(self.splits_dir, f'{split}.txt')
        else: # use the full path 
            metadata_path = os.path.join(root, os.path.expandvars(split))

        #read metadata from a txt file with no commas, but newlines
        with open(metadata_path, 'r') as f:
            self.metadata = f.read().splitlines()

        self.ds_name =  'spectral_earth'
        self.chn_ids = torch.tensor(extract_wavemus(load_ds_cfg(self.ds_name), full_spectra), dtype=torch.long)

        if self.normalize:
            self.channelwise_transforms = self._build_ch_transforms()

        self.return_multilabel = return_multilabel


    def _corine_label_43_to_19(self, label):
        return self.label_converter[self.CORINE_CLASSES_43[label]]

    def _build_ch_transforms(self):
        self.MEAN = torch.tensor(self.ENMAP_MEAN)
        self.STD = torch.tensor(self.ENMAP_STD)
        
        return transforms.Compose([
            # transforms.Normalize(self.MEAN, self.STD),
            ClipMinMaxNormalize(min_val=0.0, max_val=10000.),
            transforms.Normalize(self.MEAN, self.STD),
        ])


    def build_dataset(self, splits_dir, build_dir):
        """
        iterate over all sub directories in dir and save their paths to a txt file
        """
        splits_file = os.path.join(splits_dir, 'train.txt')
        #check if splits_file exists, if not create it
        if not os.path.exists(splits_file):
            logger.info(f"Creating new splits file: {splits_file}")
            print(f"Creating new splits file: {splits_file}")
            Path(splits_file).touch()

        with open(splits_file, 'w') as f:
            for root, _, files in os.walk(build_dir):
                for file in files:
                    if file.endswith('.tif'):
                        # f.write(os.path.join(dir, file) + '\n')
                        relative_path = os.path.relpath(os.path.join(root, file), build_dir)
                        f.write(relative_path + '\n')

    def __len__(self):
            return len(self.metadata)

    def _load_img(self, path) -> torch.Tensor:
        return torch.from_numpy(imread(path).astype('float32')).permute(2,0,1) 


    def _load_label(self, path) -> torch.Tensor:
        if not self.return_multilabel:
            return torch.from_numpy(imread(path))#.astype('float32'))
        else:
            label = imread(path)
            unique_classes = np.unique(label)
            # print(f'Unique classes: {unique_classes}')
            indices = [self._corine_label_43_to_19(c) for c in unique_classes]
            # print(f'Indices: {indices}')
            # remove any None values
            indices = [i for i in indices if i is not None]
            # print(f'Indices: {indices}')
            image_target = torch.zeros(self.num_classes, dtype=torch.long)
            image_target[indices] = 1
            return image_target


    def __getitem__(self, idx):
        # Load the image
        path = self.metadata[idx]
        img_path = os.path.join(self.data_dir, path)

        try:
            img = self._load_img(img_path)
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            print(f"Error loading image {img_path}: {e}")
            traceback.print_exc()
            img = None

        if self.labels_dir: #i.e. its a downstream task
            label_path = os.path.join(self.labels_dir, path)

            try:
                label = self._load_label(label_path)
            except Exception as e:
                logger.error(f"Error loading label {label_path}: {e}")
                # print(f"Error loading label {label_path}: {e}")
                traceback.print_exc()
                label =  None

        if self.normalize:
            # print('Normalizing image')
            img = self.channelwise_transforms(img)

        if self.return_rgb:
            chn_id = self.chn_ids
            rgb_idx = torch.tensor(self.RGB_CHANNELS[self.ds_name])
            img = img[rgb_idx]
            chn_id = chn_id[rgb_idx]
        elif self.return_chns_by_id is not None:
            idx = self.return_chns_by_id
            chn_id = self.chn_ids[idx]
            img = img[idx]
        else:
            chn_id = self.chn_ids

        if self.labels_dir: #downstream task
            sample = {"imgs": img, "chn_ids": chn_id}

            if self.transform is not None:
                sample = self.transform(sample)

            return sample, label
            
        sample = {"imgs": [img], "chn_ids": [chn_id],}
        sample = [sample] # wrapper for dino-augm
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    
    def get_targets(self):
        return np.arange(self.num_classes)



        