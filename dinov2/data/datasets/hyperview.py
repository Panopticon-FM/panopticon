import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from glob import glob
from typing import Optional, Tuple, Union
from torchvision import transforms
from dinov2.utils.data import load_ds_cfg, extract_wavemus

class Hyperview(Dataset):
    """PyTorch Dataset for loading hyperspectral datacubes with optional masking.
    
    Args:
        data_dir (str): Directory containing the .npz files with hyperspectral data
        labels_path (str): Path to the CSV file containing ground truth labels
        mask (bool): If True, applies masking where masked regions are set to -1
    """
    def __init__(
        self, 
        data_dir: str,
        labels_path: Optional[str] = None,
        mask: bool = True
    ):
        self.data_dir = data_dir
        self.mask = mask
        
        # Get sorted list of all .npz files
        self.file_paths = sorted(
            glob(os.path.join(data_dir, "*.npz")),
            key=lambda x: int(os.path.basename(x).replace(".npz", ""))
        )
        
        # Load labels if provided
        self.labels = None
        if labels_path is not None:
            gt_df = pd.read_csv(labels_path)
            self.labels = gt_df[["P", "K", "Mg", "pH"]].values
            
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Load the .npz file
        with np.load(self.file_paths[idx]) as npz:
            # Create masked array from the saved data
            data = np.ma.MaskedArray(**npz)
        
        if self.mask:
            # Get the mask (should be uniform across channels)
            mask = data.mask[0]  # Take mask from first channel
            # Convert to float32 and set masked values to -1
            data_tensor = torch.from_numpy(data.filled(-1)).float()
        else:
            # If not using mask, fill masked values with the underlying data
            data_tensor = torch.from_numpy(data.filled(data.fill_value)).float()
            
        # Return data with labels if labels exist, otherwise just return data
        if self.labels is not None:
            labels_tensor = torch.from_numpy(self.labels[idx]).float()
            return data_tensor, labels_tensor
        
        return data_tensor
    
class HyperviewWrapper(Hyperview):

    def __init__(self, 
                data_dir: str,
                labels_path: Optional[str] = None,
                mask: bool = True,
                transform=None,
                normalize: bool = True,
                partition: float = 1.,
                full_spectra: bool = False,
                ):

        super().__init__(data_dir, labels_path, mask)
        self.transform = transform

        self.ds_name = 'hyperview'
        self.chn_ids = torch.tensor(extract_wavemus(load_ds_cfg(self.ds_name), full_spectra), dtype=torch.long)
        
    def __getitem__(self, idx: int) -> torch.Tensor | Tuple[torch.Tensor]:
        sample, label =  super().__getitem__(idx)

        chn_ids = self.chn_ids
            
        sample: dict = {"imgs": sample, 
                        "chn_ids" : chn_ids,}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label
