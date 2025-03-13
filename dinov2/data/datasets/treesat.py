## Copied and modified from:
## https://github.com/gastruc/OmniSat/blob/main/src/data/TreeSAT.py

from torch.utils.data import Dataset
import h5py
import numpy as np
import json
from skmultilearn.model_selection import iterative_train_test_split
import torch
import rasterio
from datetime import datetime
from torchvision import transforms
from dinov2.utils.data import load_ds_cfg, extract_wavemus

def subset_dict_by_filename(files_to_subset, dictionary):
    return {file : dictionary[file] for file in files_to_subset}

def filter_labels_by_threshold(labels_dict, area_threshold = 0.07):
    """
    Parameters
    ----------
    labels_dict: dict, {filename1: [(label, area)],
                        filename2: [(label, area), (label, area)],
                        ...
                        filenameN: [(label, area), (label, area)]}
    area_threshold: float
    
    Returns
    -------
    filtered: dict, {filename1: [label],
                     filename2: [label, label],
                     ...
                     filenameN: [label, label]}
    """
    filtered = {}
    
    for img in labels_dict:
        for lbl, area in labels_dict[img]:
            # if area greater than threshold we keep the label
            if area > area_threshold:
                # init the list of labels for the image
                if img not in filtered:
                    filtered[img] = []
                # add only the label, since we won't use area information further
                filtered[img].append(lbl)
                
    return filtered

def collate_fn(batch):
    """
    Collate function for the dataloader.
    Args:
        batch (list): list of dictionaries with keys "label", "name"  and the other corresponding to the modalities used
    Returns:
        dict: dictionary with keys "label", "name"  and the other corresponding to the modalities used
    """
    keys = list(batch[0].keys())
    output = {}
    for key in ["s2", "s1-asc", "s1-des", "s1"]:
        if key in keys:
            idx = [x[key] for x in batch]
            max_size_0 = max(tensor.size(0) for tensor in idx)
            stacked_tensor = torch.stack([
                    torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, 0, 0, max_size_0 - tensor.size(0)))
                    for tensor in idx
                ], dim=0)
            output[key] = stacked_tensor
            keys.remove(key)
            key = '_'.join([key, "dates"])
            idx = [x[key] for x in batch]
            max_size_0 = max(tensor.size(0) for tensor in idx)
            stacked_tensor = torch.stack([
                    torch.nn.functional.pad(tensor, (0, max_size_0 - tensor.size(0)))
                    for tensor in idx
                ], dim=0)
            output[key] = stacked_tensor
            keys.remove(key)
    if 'name' in keys:
        output['name'] = [x['name'] for x in batch]
        keys.remove('name')
    for key in keys:
        output[key] = torch.stack([x[key] for x in batch])
    return output

def day_number_in_year(date_arr, place=4):
    day_number = []
    for date_string in date_arr:
        date_object = datetime.strptime(str(date_string).split('_')[place][:8], '%Y%m%d')
        day_number.append(date_object.timetuple().tm_yday) # Get the day of the year
    return torch.tensor(day_number)

def replace_nans_with_mean(batch_of_images):
    image_means = torch.nanmean(batch_of_images, dim=(3, 4), keepdim=True)
    image_means[torch.isnan(image_means)] = 0.
    nan_mask = torch.isnan(batch_of_images)
    batch_of_images[nan_mask] = image_means.expand_as(batch_of_images)[nan_mask]
    return batch_of_images

class TreeSAT(Dataset):

    classes = ["Abies",
            "Acer",
            "Alnus",
            "Betula",
            "Cleared",
            "Fagus",
            "Fraxinus",
            "Larix",
            "Picea",
            "Pinus",
            "Populus",
            "Prunus",
            "Pseudotsuga",
            "Quercus",
            "Tilia"]
    
    def __init__(
        self,
        path,
        modalities,
        split: str = "train",
        partition: float = 1.,
        mono_strict: bool = False,
        ):
        """
        Initializes the dataset.
        Args:
            path (str): path to the dataset
            modalities (list): list of modalities to use; for now:
                ['aerial', 's2-mono', 's1-mono']
            transform (torch module): transform to apply to the data
            split (str): split to use (train, val, test)
            classes (list): name of the differrent classes
            partition (float): proportion of the dataset to keep
            mono_strict (bool): if True, puts monodate in same condition as multitemporal
        """
        self.path = path
        self.partition = partition
        self.modalities = modalities
        self.mono_strict = mono_strict
        data_path = path + split + "_filenames.lst"
        print(f'Data path: {data_path}')
        with open(data_path, 'r') as file:
            self.data_list = [line.strip() for line in file.readlines()]


        self.load_labels(self.classes)
        self.collate_fn = collate_fn
            
    def load_labels(self, classes):
        with open(self.path + "labels/TreeSatBA_v9_60m_multi_labels.json") as file:
            jfile = json.load(file)
            subsetted_dict = subset_dict_by_filename(self.data_list, jfile)
            labels = filter_labels_by_threshold(subsetted_dict, 0.07)
            lines = list(labels.keys())

        y = [[0 for i in range(len(classes))] for line in lines]
        for i, line in enumerate(lines):
            for u in labels[line]:
                y[i][classes.index(u)] = 1

        self.data_list, self.labels, _, _ = iterative_train_test_split(np.expand_dims(np.array(lines), axis=1), np.array(y), test_size = 1. - self.partition)
        self.data_list = list(np.concatenate(self.data_list).flat)

    def __getitem__(self, i):
        """
        Returns an item from the dataset.
        Args:
            i (int): index of the item
        Returns:
            dict: dictionary with keys "label", "name" and the other corresponding to the modalities used
        """
        name = self.data_list[i]
        output = {'label': torch.tensor(self.labels[i])}
    
        if 'aerial' in self.modalities:
            with rasterio.open(self.path + "aerial/" + name) as f:
                output["aerial"] = torch.FloatTensor(f.read())

        # with h5py.File(self.path + "sentinel-ts/" + '.'.join(name.split('.')[:-1]) + ".h5", 'r') as file:
        #     if 's1-asc' in self.modalities:
        #         output["s1-asc_dates"] = day_number_in_year(file["sen-1-asc-products"][:])
        #     if 's1-des' in self.modalities:
        #         output["s1-des_dates"] = day_number_in_year(file["sen-1-des-products"][:])
        #     if 's1' in self.modalities:
        #         s1_asc_dates = day_number_in_year(file["sen-1-asc-products"][:])
        #         s1_des_dates = day_number_in_year(file["sen-1-des-products"][:])
        #     if 's2' in self.modalities:
        #         output["s2"]= torch.tensor(file["sen-2-data"][:])
        #         output["s2_dates"] = day_number_in_year(file["sen-2-products"][:], place=2)
        #         N = len(output["s2_dates"])
        #         if N > 50:
        #             random_indices = torch.randperm(N)[:50]
        #             output["s2"] = output["s2"][random_indices]
        #             output["s2_dates"] = output["s2_dates"][random_indices]

        if 's1-asc' in self.modalities:
            output["s1-asc"] = torch.load(self.path + "s1-asc/" + '.'.join(name.split('.')[:-1]) + ".pth")[:, :2, : ,:]
            N = len(output["s1-asc_dates"])
            if N > 50:
                random_indices = torch.randperm(N)[:50]
                output["s1-asc"] = output["s1-asc"][random_indices]
                output["s1-asc_dates"] = output["s1-asc_dates"][random_indices]

        if 's1-des' in self.modalities:
            output["s1-des"] = torch.load(self.path + "s1-des/" + '.'.join(name.split('.')[:-1]) + ".pth")[:, :2, : ,:]
            N = len(output["s1-des_dates"])
            if N > 50:
                random_indices = torch.randperm(N)[:50]
                output["s1-des"] = output["s1-des"][random_indices]
                output["s1-des_dates"] = output["s1-des_dates"][random_indices]

        if 's1' in self.modalities:
            s1_asc = torch.load(self.path + "s1-asc/" + '.'.join(name.split('.')[:-1]) + ".pth")[:, :2, : ,:]
            s1_des = torch.load(self.path + "s1-des/" + '.'.join(name.split('.')[:-1]) + ".pth")[:, :2, : ,:]
            output["s1"] = torch.cat([s1_asc, s1_des], dim=0)
            output["s1_dates"] = torch.cat([s1_asc_dates, s1_des_dates], dim=0)
            N = len(output["s1_dates"])
            if N > 50:
                random_indices = torch.randperm(N)[:50]
                output["s1"] = output["s1"][random_indices]
                output["s1_dates"] = output["s1_dates"][random_indices]

        if "s1-mono" in self.modalities:
            with rasterio.open(self.path + "s1/60m/" + name) as f:
                numpy_array = f.read()
            numpy_array = numpy_array.astype(np.float32)
            output["s1-mono"] = torch.FloatTensor(numpy_array)
            if self.mono_strict:
                output["s1-mono"] = output["s1-mono"][:2, :, :]
            # remove the band ratio band
            output["s1-mono"] = output["s1-mono"][:2, :, :]

        if "s2-mono" in self.modalities:
            with rasterio.open(self.path + "s2/60m/" + name) as f:
                numpy_array = f.read()
            numpy_array = numpy_array.astype(np.float32)
            output["s2-mono"] = torch.FloatTensor(numpy_array)
            if self.mono_strict:
                output["s2-mono"] = output["s2-mono"][:10, :, :]

        if "s2-4season-median" in self.modalities:
            with h5py.File(self.path + "sentinel/" + '.'.join(name.split('.')[:-1]) + ".h5", 'r') as file:
                output_inter = torch.tensor(file["sen-2-data"][:])
                dates = day_number_in_year(file["sen-2-products"][:], place=2)
            l = []
            for i in range (4):
                mask = ((dates >= 92 * i) & (dates < 92 * (i+1)))
                if sum(mask) > 0:
                    r, _ = torch.median(output_inter[mask], dim = 0)
                    l.append(r)
                else:
                    l.append(torch.zeros((output_inter.shape[1], output_inter.shape[-2], output_inter.shape[-1])))
            output["s2-4season-median"] = torch.cat(l)

        if "s2-median" in self.modalities:
            with h5py.File(self.path + "sentinel/" + '.'.join(name.split('.')[:-1]) + ".h5", 'r') as file:
                output["s2-median"], _ = torch.median(torch.tensor(file["sen-2-data"][:]), dim = 0)

        if "s1-4season-median" in self.modalities:
            with h5py.File(self.path + "sentinel/" + '.'.join(name.split('.')[:-1]) + ".h5", 'r') as file:
                dates = day_number_in_year(file["sen-1-asc-products"][:])
            output_inter = torch.load(self.path + "s1-asc/" + '.'.join(name.split('.')[:-1]) + ".pth")[:, :2, : ,:]
            l = []
            for i in range (4):
                mask = ((dates >= 92 * i) & (dates < 92 * (i+1)))
                if sum(mask) > 0:
                    r, _ = torch.median(output_inter[mask], dim = 0)
                    l.append(r)
                else:
                    l.append(torch.zeros((output_inter.shape[1], output_inter.shape[-2], output_inter.shape[-1])))
            output["s1-4season-median"] = torch.cat(l)

        if "s1-median" in self.modalities:
            output["s1-median"], _ = torch.median(torch.load(self.path + "s1-asc/" + '.'.join(name.split('.')[:-1]) + ".pth")[:, :2, : ,:], dim = 0)

        return output

    def __len__(self):
        return len(self.data_list)

class TreeSATWrapper(TreeSAT):
    #means and stds from:
    # https://git.tu-berlin.de/rsim/treesat_benchmark/-/blob/master/configs/multimodal/AllModes_Xformer_ResnetScratch_v8.json?ref_type=heads
    # 

    MEAN: dict = { "s2-mono" : torch.tensor([231.43385024546893, 376.94788434611434, 241.03688288984037, 2809.8421354087955, 616.5578221193639, 2104.3826773960823, 2695.083864757169, 2969.868417923599, 1306.0814241837832, 587.0608264363341, 249.1888624097736, 2950.2294375352285]),
                   "s1-mono" : torch.tensor([-6.933713050794077, -12.628564056094067]),
                   "aerial" : torch.tensor([151.26809261440323, 93.1159469148246, 85.05016794624635, 81.0471576353153])
    }
    STDEV : dict = {"s2-mono" : torch.tensor([123.16515044781909, 139.78991338362886, 140.6154081184225, 786.4508872594147, 202.51268536579394, 530.7255451201194, 710.2650071967689, 777.4421400779165, 424.30312334282684, 247.21468849049668, 122.80062680549261, 702.7404237034002]),
                    "s1-mono" : torch.tensor([87.8762246957811, 47.03070478433704]),
                    "aerial" : torch.tensor([48.70879149145466, 33.59622314610158, 28.000497087051126, 33.683983599997724])
    }
  
    def _build_ch_transforms(self):
        channelwise_transforms = {}
        for sensor in self.MEAN.keys():
            channelwise_transforms[sensor] = transforms.Normalize(self.MEAN[sensor], self.STDEV[sensor])
        return channelwise_transforms

    def __init__(self,
                path,
                modalities,
                split: str = "train",
                transform=None,
                normalize: bool = True,
                partition: float = 1.,):
        super(TreeSATWrapper, self).__init__(path, modalities, split, partition)

        self.sensor_name_mapping = {'s1-mono': 'treesat-s1', 's2-mono': 'treesat-s2', 'aerial': 'treesat-aerial'}
        self.channelwise_transforms = self._build_ch_transforms()
        self.normalize = normalize
        self.chn_ids = {k: extract_wavemus(load_ds_cfg(v)) for k, v in self.sensor_name_mapping.items() }
        self.transform = transform


    def __getitem__(self, idx):
        sample = super(TreeSATWrapper, self).__getitem__(idx)

        labels = sample.pop('label')

        # listify the images
        imgs = [self.channelwise_transforms[sensor](sample[sensor]) if self.normalize else sample[sensor] for sensor in sample]
        chn_ids = [self.chn_ids[sensor] for sensor in sample if sensor in sample]

        out = dict(
            imgs = [[i] for i in imgs], 
            chn_ids = chn_ids,
            # times = times)  
        )

        if self.transform:
            out = self.transform(out)
        return out, labels
    