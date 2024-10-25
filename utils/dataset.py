import os
import json
import re
from PIL import Image, ImageOps
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

####################################################################################################################################################################################

class cus_Dataset(Dataset):

    def __init__(self, csv_path, transform):

        csv_name = os.path.basename(csv_path)
        csv_name, _ = os.path.splitext(csv_name)
        parts = csv_name.split('_')

        self.df = pd.read_csv(csv_path)
        self.data_path = '/home/ubuntu/workspace/dataset/084.화상 이미지 및 임상 데이터/01-1.정식개방데이터'

        # csv_name : 5classes_all_Training_Single.csv
        # mode : Training / Validation
        # shape : Single / Time series
        mode = parts[2]
        shape = parts[3]
        self.image_dir = os.path.join(self.data_path, mode, '01.원천데이터', shape)
        self.label_dir = os.path.join(self.data_path, mode, '02.라벨링데이터', shape)
        
        self.transform = transform


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):

        image = self.load_image(idx)
        label = self.load_label(idx)

        if self.tranform is not None:
            image = self.transform(image)

        return image, label
    

    def load_image(self, idx):

        # bbox : (x, y, wwidth, height)
        bbox = re.findall(r'\d+', self.df['bbox'][idx])
        bbox = [int(x) for x in bbox]
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]

        try:
            img = Image.open(self.df['image_path'][idx]).convert('RGB')
            img = ImageOps.exif_transpose(img)
            img = img.crop((x1, y1, x2, y2))
            return img
        
        except Exception as e:
            random_arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            random_img = Image.fromarray(random_arr)
            return random_img
        

    def load_label(self, idx):

        label = self.df['label'][idx]

        return label