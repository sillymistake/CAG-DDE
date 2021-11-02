import glob
import math
import random
import numbers
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image


mean = np.array([0.485, 0.456, 0.406]).reshape([1,1,3])
norm = np.array([0.229, 0.224, 0.225]).reshape([1,1,3])

class MyDataset_TR(Dataset):

    def __init__(self, img_list, is_origin=False, transform=None):
        self.img_list = img_list
        self.is_origin = is_origin
        self.transform = transform
        self.len = len(self.img_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image = Image.open(self.img_list[idx]['img']).convert('RGB') # (w,h)
        flow = Image.open(self.img_list[idx]['flo']).convert('RGB')
        label = Image.open(self.img_list[idx]['lbl']).convert('L')

        assert image.size == label.size
        assert image.size == flow.size

        if not self.is_origin:
            if isinstance(self.transform['size'], numbers.Number):
                img_size = (int(self.transform['size']), int(self.transform['size']))
            else:
                img_size = self.transform['size'] # (w,h)
            image = image.resize(img_size, Image.BILINEAR)
            flow = flow.resize(img_size, Image.BILINEAR)
            label = label.resize(img_size, Image.NEAREST)

        if self.transform['flip']:
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                flow = flow.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform['rotate']:
            alpha = random.randint(-30, 30)
            image = image.rotate(alpha)
            flow = flow.rotate(alpha)
            label = label.rotate(alpha)

        image = np.array(image).astype(np.float32) # (h,w,c)
        flow = np.array(flow).astype(np.float32)
        label = np.array(label)
        
        if self.transform['jitter']:
            scale = random.random() + 0.5
            light = random.randint(-30, 30)
            image = scale * image + light
            image = np.clip(image, 0, 255)
            flow = scale * flow + light
            flow = np.clip(flow, 0, 255)

        image = ((image / 255.0) - mean) / norm
        image = image.transpose((2, 0, 1)) # (c,h,w)
        flow = ((flow / 255.0) - mean) / norm
        flow = flow.transpose((2, 0, 1))
        label = (label > 127).astype('float32')
        label = np.expand_dims(label, 0) # (1,h,w)

        # image & flow: c,h,w   label: 1,h,w
        sample = {'img':image, 'flo':flow, 'lbl':label}

        return sample

class MyDataset_TE(Dataset):
    def __init__(self, img_list, is_origin=False, transform=None):
        self.img_list = img_list
        self.is_origin = is_origin
        self.transform = transform
        self.len = len(self.img_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        name = self.img_list[idx]['lbl']
        image = Image.open(self.img_list[idx]['img']).convert('RGB')
        flow = Image.open(self.img_list[idx]['flo']).convert('RGB')
        label = Image.open(self.img_list[idx]['lbl']).convert('L')

        # image without dataloader
        if not self.is_origin:
            if isinstance(self.transform['size'], numbers.Number):
                img_size = (int(self.transform['size']), int(self.transform['size']))
            else:
                img_size = self.transform['size'] # (w,h)
            image = image.resize(img_size, Image.BILINEAR)
            flow = flow.resize(img_size, Image.BILINEAR)

        image = np.array(image).astype(np.float32)
        image = ((image / 255.0) - mean) / norm
        image = image.transpose((2, 0, 1))

        flow = np.array(flow).astype(np.float32)
        flow = ((flow / 255.0) - mean) / norm
        flow = flow.transpose((2, 0, 1))

        # original label
        label = np.array(label)
        label = (label > 127).astype('float32')

        sample = {'name':name, 'img':image, 'flo':flow, 'lbl':label}

        return sample


if __name__ == '__main__':
    from load_data import *
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    tra_list, tes_list = load_davis()
    transform_tra = {'size':(448,224), 'flip':True, 'rotate':True, 'jitter':True}
    transform_tes = {'size':(448,224)}
    my_dataset = MyDataset_TR(tra_list, is_origin=False, transform=transform_tra)
    my_dataloader = DataLoader(my_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    train_iter = tqdm(my_dataloader, ncols=150)
    for i, data in enumerate(train_iter, start=1):
        image = data['img']
        flow = data['flo']
        label = data['lbl']
        print(image.shape)
        print(flow.shape)
        print(label.shape)
        break

    test_dataset = MyDataset_TE(tes_list, is_origin=False, transform=transform_tes)
    sample = test_dataset.load_data(0)
    image = sample['img']
    flow = sample['flo']
    label = sample['lbl']
    print(image.shape)
    print(flow.shape)
    print(label.shape)




