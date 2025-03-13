
import os
import pandas as pd
from torch.utils.data import Dataset
import torch
import tifffile as tiff
import numpy as np

from dinov2.utils.data import extract_wavemus, load_ds_cfg

# with minor adjustments from https://github.com/zhu-xlab/FGMAE/blob/main/src/transfer_classification/datasets/EuroSat/eurosat_dataset_s1.py


class EurosatSAR(Dataset):

    # from FGMAE, probably computed after quantile normalization
    MEAN_QUANTILE = [-12.59, -20.26]
    STD_QUANTILE = [5.26, 5.91]

    # computed on train split only
    MEAN = [-11.6568, -18.4975]
    STD = [6.0908, 6.1924]

    class_to_idx = {
        'AnnualCrop': 0,
        'Forest': 1,
        'HerbaceousVegetation': 2,
        'Highway': 3,
        'Industrial': 4,
        'Pasture': 5,
        'PermanentCrop': 6,
        'Residential': 7,
        'River': 8,
        'SeaLake': 9
    }


    def __init__(self, root, full_spectra = False, normalize = True, transform = None, split = 'train'):
        self.root = root
        self.normalize = normalize
        self.transform = transform
        self.chn_ids = extract_wavemus(load_ds_cfg('eurosat-sar'), return_sigmas=full_spectra)
        self.num_classes = 10

        self.df = pd.concat([pd.read_csv(os.path.join(root, f'{s}.csv')) 
                             for s in split.split(',')], ignore_index=True)
        self.df['class_idx'] = self.df['class_name'].map(self.class_to_idx)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        filepath = os.path.join(self.root, self.df.loc[idx, 'filepath'])
        x_dict = self._load_img(filepath)
        label = self.df.loc[idx, 'class_idx']
        return x_dict, label

    def _load_img(self, path):
        img = tiff.imread(path)

        if self.normalize:
            img = (img - self.MEAN) / self.STD

        img = torch.from_numpy(img).permute(2, 0, 1).to(dtype=torch.float32)
        x_dict = dict(imgs=img, chn_ids=self.chn_ids)

        if self.transform is not None:
            x_dict = self.transform(x_dict)

        return x_dict

    def _normalize_to_uint8(self, img, mean, std):
        """ not needed """

        chns = []

        for i in range(2):
            ch = img[:,:,i]

            max_q = np.quantile(ch, 0.99)      
            min_q = np.quantile(ch, 0.01)            
            ch[ch > max_q] = max_q
            ch[ch < min_q] = min_q
            
            mean = self.MEAN_QUANTILE[i]
            std = self.STD_QUANTILE[i]

            min_value = mean - 2 * std
            max_value = mean + 2 * std
            ch = (ch - min_value) / (max_value - min_value) * 255.0
            ch = np.clip(ch, 0, 255).astype(np.uint8)

            chns.append(ch)

        img = np.stack(chns, axis=2)
        return img


if __name__ == '__main__':

    """ create splits in a reproducible way"""

    import pandas as pd
    import os
    from sklearn.model_selection import train_test_split

    # get full df

    base_dir = '/hkfs/work/workspace/scratch/tum_mhj8661-panopticon/datasets/eurosat_SAR'

    all_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.tif'):
                filepath = os.path.relpath(os.path.join(root, file), base_dir)
                class_name = os.path.basename(root)
                all_files.append((filepath, class_name))

    df = pd.DataFrame(all_files, columns=['filepath', 'class_name'])
    df = df.sort_values('filepath') # make unique regardles of os.walk order

    # sample all classes to same number

    min_class_samples = df['class_name'].value_counts().min()
    out_df = []

    for i,c in enumerate(df['class_name'].unique()):
        _df = df[df['class_name'] == c]
        if _df.shape[0] > min_class_samples:
            _df = _df.sample(min_class_samples, random_state=i)
        out_df.append(_df)

    df = pd.concat(out_df)

    # stratified split

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['class_name'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=1/8, stratify=train_df['class_name'], random_state=42)

    small_train_df, small_test_df = train_test_split(train_df, train_size=3000, test_size=1000, stratify=train_df['class_name'], random_state=42)
    small_train_df, small_val_df = train_test_split(small_train_df, test_size=1000, stratify=small_train_df['class_name'], random_state=42)

    # save

    names = dict(
        train = train_df,
        val = val_df,
        test = test_df,
        small_train = small_train_df,
        small_val = small_val_df,
        small_test = small_test_df
        )

    names = {k: v.sort_values('filepath').reset_index(drop=True) for k,v in names.items()}
    for name, df in names.items():
        print('--------------------------')
        print(name, df.shape[0])
        print(df['class_name'].value_counts())
    df.to_csv(os.path.join(base_dir, f'{name}.csv'), index=False)