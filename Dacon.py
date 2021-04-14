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
import cv2
from torchvision.transforms import transforms

class Dacon(Dataset):
    def __init__(self):
        with open('zero.json', 'r') as f:
            zero_list = json.load(f)
        self.image_size = (288, 384)#(1920, 1080)#(1080, 920)
        self.image_list = [str(x) for x in Path('bbox_info').iterdir() if 'img' in x.name]
        self.bbox_list = [str(x) for x in Path('bbox_info').iterdir() if 'bbox' in x.name]

        self.image_list.sort()
        self.bbox_list.sort()
        self.nof_joints = 17

        for i in zero_list:
            self.image_list.remove(i)
            self.bbox_list.remove(i.replace('_img', '_bbox'))
            
        with open('labels.json', 'r') as f:
            self.labels = json.load(f)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((384, 288)),  # (height, width)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_list)


    
    def _generate_target(self, joints):

        target = np.zeros((self.nof_joints,
                            96,
                            72),
                            dtype=np.float32)

        tmp_size = 3 * 3

        for joint_id in range(self.nof_joints):
            feat_stride = np.asarray(self.image_size) / np.asarray((72, 96))
            
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)

            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * 3 ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], 72) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], 96) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], 72)
            img_y = max(0, ul[1]), min(br[1], 96)
            
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target



    def __getitem__(self, idx):
        
        img_path = self.image_list[idx].replace('bbox_info', '../Dacon/data/train_imgs').replace('_img.pkl', '.jpg')

        data = cv2.imread(img_path, cv2.IMREAD_COLOR)
        lab= cv2.cvtColor(data, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        with open(self.bbox_list[idx], 'rb') as f:
            bbox = pickle.load(f)
        
        if self.image_list[idx][:35] != self.bbox_list[idx][:35]:
            pdb.set_trace()

        
        fname = self.image_list[idx].split('_img.pkl')[0].split('bbox_info/')[1] + '.jpg'
        x1, y1, x2, y2 = bbox[0]
        
        labels = np.array(self.labels[fname])
        if x2 < max(np.asarray(labels)[:,0]):
            x2 = int(max(np.asarray(labels)[:,0]))

        if y2 < max(np.asarray(labels)[:,1]):
            y2 = int(max(np.asarray(labels)[:,1]))

        if x1 > min(np.asarray(labels)[:,0]):
            x1 = int(min(np.asarray(labels)[:,0]))

        if y1 > min(np.asarray(labels)[:,1]):
            y1 = int(min(np.asarray(labels)[:,1]))

        img = self.transform(img[y1:y2, x1:x2, ::-1])
        bbox = np.array([x1, y1, x2, y2])
        new_labels = labels[:17].copy()

        new_labels[:,0] = (new_labels[:,0]-x1)*288 / (x2-x1)
        new_labels[:,1] = (new_labels[:,1]-y1)*384 / (y2-y1)

        try:
            target_heat = self._generate_target(new_labels)
        except:
            pdb.set_trace()


        return img.squeeze(), bbox, target_heat, self.image_list[idx], new_labels.copy()#labels[:17, ::-1].copy()
