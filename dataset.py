import os

import pandas as pd
import torch
from PIL import Image


class CardDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, csv_path, transform):

        cols = [
            'label_name',
            'rect_left',
            'rect_top',
            'rect_width',
            'rect_height',
            'image_name',
            'image_width',
            'image_height',
        ]

        self.class_to_id = {
            'mastercard': 0,
            'visa': 1,
        }

        self.image_folder = image_folder
        self.csv_path = csv_path

        df = pd.read_csv(csv_path, header=None, names=cols)
        df['rect_left_rel'] = df['rect_left'] / df['image_width']
        df['rect_top_rel'] = df['rect_top'] / df['image_height']
        df['rect_width_rel'] = df['rect_width'] / df['image_width']
        df['rect_height_rel'] = df['rect_height'] / df['image_height']

        self.df = df
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_folder, row['image_name'])
        img = Image.open(img_path).convert('RGBA').convert('RGB')
        if self.transform:
            img = self.transform(img)

        label = self.class_to_id[row['label_name']]
        xmin_rel = row['rect_left'] / row['image_width']
        xmax_rel = (row['rect_left'] + row['rect_width']) / row['image_width']
        ymin_rel = (row['rect_top'] - row['rect_height']) / row['image_height']
        ymax_rel = row['rect_top'] / row['image_height']

        return img, label, (xmin_rel, xmax_rel, ymin_rel, ymax_rel)

    def __len__(self):
        return self.df.shape[0]
