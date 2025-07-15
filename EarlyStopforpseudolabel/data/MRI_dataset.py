import os
from io import BytesIO
import lmdb
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import nibabel as nib


class MRIDataset(Dataset):
    def __init__(self, dataroot, data_sequence, datatype, nclass, split='train', data_len=-1):
        self.datatype = datatype
        self.data_sequence = data_sequence
        self.data_len = data_len
        self.split = split
        self.nclass = nclass
        
        self.ori_path = Util.get_paths_from_images(dataroot, data_sequence['cond_image'])
        self.tg_path = Util.get_paths_from_images(dataroot, data_sequence['output_label'])
        # self.pl_path = Util.get_paths_from_images(dataroot, data_sequence['cond_pesudolabel'])
        
        self.dataset_len = len(self.ori_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)
            

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        
        ori = np.load(self.ori_path[index])
        tg = np.load(self.tg_path[index])
        
        [ori, tg] = Util.transform_augment([ori, tg], self.nclass, split=self.split, min_max=(-1, 1))
        
        return {'original': torch.cat([ori,ori,ori], dim=0),
                'target': tg, # 'pesudolabel': pl, 
                'Index': index}
