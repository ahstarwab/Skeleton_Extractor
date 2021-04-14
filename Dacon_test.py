import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys
from pathlib import Path
import pickle
import pandas as pd
import json
import pdb


class Dacon(Dataset):
    def __init__(self):
        with open('zero_test.json', 'r') as f:
            zero_list = json.load(f)
        self.image_size = (1920, 1080)#(1080, 920)
        self.image_list = [str(x) for x in Path('bbox_test').iterdir() if 'img' in x.name]
        self.bbox_list = [str(x) for x in Path('bbox_test').iterdir() if 'bbox' in x.name]

        self.image_list.sort()
        self.bbox_list.sort()
        self.nof_joints = 17

        for i in zero_list:
            self.image_list.remove(i)
            self.bbox_list.remove(i.replace('_img', '_bbox'))
            
        with open('labels.json', 'r') as f:
            self.labels = json.load(f)
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        with open(self.image_list[idx], 'rb') as f:
            img = pickle.load(f)

        with open(self.bbox_list[idx], 'rb') as f:
            bbox = pickle.load(f)

        if img.size(0) > 1:
            img = img[0]

        if img.size(0) == 0:
            pdb.set_trace()

        return img.squeeze(), bbox[0]