import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import nibabel as nib
from .mask import (custom_uniform_downsampling_mask,Gen_Sampling_Mask,get_mask)

def convert_to_k_space(mri_image):
    # 进行二维FFT，得到K-space数据
    k_space = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(mri_image)))
    return k_space

def k_space_to_img(k_space):
    #进行逆傅里叶变化得到图片
    img = np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(k_space))))
    return img

def normalize_npy(img):
    min_val = np.min(img)
    max_val = np.max(img)
    img = (img - min_val) / (max_val - min_val)
    return img

def make_dataset(dir):
    file_path = []
    assert os.path.isdir(dir), f'{dir} 不是一个有效的目录'
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            file_path.append(path)
    return file_path


class MRI_Restoration(data.Dataset):
    def __init__(self, data_root,acc_factor=-1, image_size=[256, 256]):
        if acc_factor!=-1:
            self.acc_factor=acc_factor
        else:
            self.acc_factor=-1
        self.paths = make_dataset(data_root)
        self.image_size = image_size

    def __getitem__(self, index):
        if self.acc_factor==-1:
            acc_factor = random.choice([4, 8, 12])
        else:
            acc_factor=self.acc_factor
        ret = {}
        path=self.paths[index]

        img = np.load(path).astype(np.float32)
        img_kspace=convert_to_k_space(img)

        mask=get_mask(img_kspace.shape)
        under_kspace=img_kspace * mask
        cond_img=k_space_to_img(under_kspace).astype(np.float32)

        img = normalize_npy(img)
        cond_img=normalize_npy(cond_img)

        gt_img = img / np.max(img)
        k_full = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(gt_img)))
        k_sub = k_full * mask.astype(np.float32)

        #k_sub=under_kspace

        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, dim=0)
        cond_img = torch.from_numpy(cond_img)
        cond_img = torch.unsqueeze(cond_img, dim=0)

        ret['gt_image'] = img
        ret['cond_image'] = cond_img
        ret['sub_kspace']=k_sub
        ret['mask']=mask
        # 提取文件名
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.paths)











