# Modified with minor adjustments from https://github.com/microsoft/torchgeo/blob/main/torchgeo/datasets/resisc45.py


# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""RESISC45 dataset."""

import os
from collections.abc import Callable
from typing import ClassVar, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from torch import Tensor
from pathlib import Path
from torch.utils.data import Dataset
from dinov2.utils.data import extract_wavemus, load_ds_cfg
import torch
from torchvision.datasets import ImageFolder
from torchvision.io import read_image

class RESISC45(Dataset):
    """NWPU-RESISC45 dataset.

    The `RESISC45 <https://doi.org/10.1109/jproc.2017.2675998>`__
    dataset is a dataset for remote sensing image scene classification.

    Dataset features:

    * 31,500 images with 0.2-30 m per pixel resolution (256x256 px)
    * three spectral bands - RGB
    * 45 scene classes, 700 images per class
    * images extracted from Google Earth from over 100 countries
    * images conditions with high variability (resolution, weather, illumination)

    Dataset format:

    * images are three-channel jpgs

    Dataset classes:

    0. airplane
    1. airport
    2. baseball_diamond
    3. basketball_court
    4. beach
    5. bridge
    6. chaparral
    7. church
    8. circular_farmland
    9. cloud
    10. commercial_area
    11. dense_residential
    12. desert
    13. forest
    14. freeway
    15. golf_course
    16. ground_track_field
    17. harbor
    18. industrial_area
    19. intersection
    20. island
    21. lake
    22. meadow
    23. medium_residential
    24. mobile_home_park
    25. mountain
    26. overpass
    27. palace
    28. parking_lot
    29. railway
    30. railway_station
    31. rectangular_farmland
    32. river
    33. roundabout
    34. runway
    35. sea_ice
    36. ship
    37. snowberg
    38. sparse_residential
    39. stadium
    40. storage_tank
    41. tennis_court
    42. terrace
    43. thermal_power_station
    44. wetland

    This dataset uses the train/val/test splits defined in the "In-domain representation
    learning for remote sensing" paper:

    * https://arxiv.org/abs/1911.06721

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/jproc.2017.2675998
    """

    url = 'https://hf.co/datasets/torchgeo/resisc45/resolve/a826b44d938a883185f11ebe3d512d38b464312f/NWPU-RESISC45.zip'
    md5 = '75206b2e16446591afa88e2628744886'
    filename = 'NWPU-RESISC45.zip'
    directory = 'NWPU-RESISC45'

    splits = ('train', 'val', 'test')
    split_urls: ClassVar[dict[str, str]] = {
        'train': 'https://hf.co/datasets/torchgeo/resisc45/resolve/a826b44d938a883185f11ebe3d512d38b464312f/resisc45-train.txt',
        'val': 'https://hf.co/datasets/torchgeo/resisc45/resolve/a826b44d938a883185f11ebe3d512d38b464312f/resisc45-val.txt',
        'test': 'https://hf.co/datasets/torchgeo/resisc45/resolve/a826b44d938a883185f11ebe3d512d38b464312f/resisc45-test.txt',
    }
    split_md5s: ClassVar[dict[str, str]] = {
        'train': 'b5a4c05a37de15e4ca886696a85c403e',
        'val': 'a0770cee4c5ca20b8c32bbd61e114805',
        'test': '3dda9e4988b47eb1de9f07993653eb08',
    }

    classes = [
        'airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach',
        'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud',
        'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway',
        'golf_course', 'ground_track_field', 'harbor', 'industrial_area', 'intersection',
        'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park',
        'mountain', 'overpass', 'palace', 'parking_lot', 'railway',
        'railway_station', 'rectangular_farmland', 'river', 'roundabout', 'runway',
        'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium',
        'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station', 'wetland'
    ]
    
    MEAN = [0.3682, 0.3808, 0.3434]
    STD = [0.2033, 0.1852, 0.1846]

    @classmethod
    def label_to_idx(cls, label: str) -> int:
        """Convert a string label to its corresponding class index."""
        return cls.classes.index(label)

    def __init__(
        self,
        root,
        split = 'train',
        normalize = True,
        transform = None,
        full_spectra = False,
    ) -> None:
        """Initialize a new RESISC45 dataset instance."""
        assert split in self.splits
        
        self.root = root
        self.normalize = normalize
        self.transform = transform
        self.chn_ids = extract_wavemus(load_ds_cfg('resisc45'), return_sigmas=full_spectra)
        self.num_classes = 45
        
        # Store base path and read valid filenames into a list
        self.base_path = os.path.join(self.root, self.directory)
        self.valid_fns = []
        with open(os.path.join(self.root, f'resisc45-{split}.txt')) as f:
            self.valid_fns = [line.strip() for line in f]

        self.labels = [self._get_label(fn) for fn in self.valid_fns]
        
        self.MEAN = torch.tensor(self.MEAN).view(3, 1, 1)
        self.STD = torch.tensor(self.STD).view(3, 1, 1)

    def _load_img(self, path):
        # Read image directly to tensor - fastest method
        img = read_image(path).to(dtype=torch.float32)

        if self.normalize:
            img = img / 255.0
            img = (img - self.MEAN) / self.STD

        x_dict = dict(imgs=img, chn_ids=self.chn_ids)

        if self.transform is not None:
            x_dict = self.transform(x_dict)

        return x_dict
    
    def _get_label(self, filename):
        # Split on last underscore and take everything before it
        label = '_'.join(filename.split('_')[:-1])
        # Remove .jpg extension if present
        label = label.replace('.jpg', '')
        return label

    def __getitem__(self, idx):
        filename = self.valid_fns[idx]
        label = self.labels[idx]
        filepath = os.path.join(self.base_path, label,filename)
        
        return self._load_img(filepath), self.label_to_idx(label)

    def __len__(self):
        return len(self.valid_fns)

    def plot(
        self,
        sample: tuple[dict[str, Tensor], str],
        show_titles: bool = True,
        suptitle: str = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by __getitem__ containing (x_dict, label)
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        x_dict, label = sample
        image = np.rollaxis(x_dict['imgs'].numpy(), 0, 3)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis('off')
        if show_titles:
            title = f'Label: {label}'
            if 'prediction' in x_dict:
                title += f'\nPrediction: {x_dict["prediction"]}'
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig