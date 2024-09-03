# We will use this file to create a dataloader for the real and fake dataset
import os
import json
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import pandas as pd
import cv2

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

class Extracted_Frames_Dataset(Dataset):
    def __init__(self, root_dir, split = "train", transform = None, extend = 'None', multi_modal = "dct"):
        """
        Args:   
        returns:
            """
        AssertionError(split in ["train", "val", "test"]), "Split must be one of (train, val, test)"
        self.multi_modal = multi_modal
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        if extend == 'faceswap':
            self.dataset = pd.read_csv(os.path.join(root_dir, f"faceswap_extended_{self.split}.csv"))
        elif extend == 'fsgan':
            self.dataset = pd.read_csv(os.path.join(root_dir, f"fsgan_extended_{self.split}.csv"))
        else:
            self.dataset = pd.read_csv(os.path.join(root_dir, f"{self.split}.csv"))
                

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample_input = self.get_sample_input(idx)
        return sample_input
    

    def get_sample_input(self, idx):
        rgb_image = self.get_rgb_image(idx)
        label = self.get_label(idx) 
        if self.multi_modal == "dct":
            dct_image = self.get_dct_image(idx)
            sample_input = {"rgb_image": rgb_image, "dct_image": dct_image, "label": label}

        # dct_image = self.get_dct_image(idx)
        elif self.multi_modal == "fft":
            fft_image = self.get_fft_image(idx)
            sample_input = {"rgb_image": rgb_image, "dct_image": fft_image, "label": label}
        elif self.multi_modal == "hh":
            hh_image = self.get_hh_image(idx)
            sample_input = {"rgb_image": rgb_image, "dct_image": hh_image, "label": label}
        else:
            AssertionError("multi_modal must be one of (dct:discrete cosine transform, fft: fast forier transform, hh)")

        return sample_input

    
    def get_fft_image(self, idx):
        gray_image_path = self.dataset.iloc[idx, 0]
        gray_image = cv2.imread(gray_image_path, cv2.IMREAD_GRAYSCALE)
        fft_image = self.compute_fft(gray_image)
        if self.transform:
            fft_image = self.transform(fft_image)
        
        return fft_image

    
    def compute_fft(self, image):
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # Add 1 to avoid log(0)
        return magnitude_spectrum


    def get_hh_image(self, idx):
        gray_image_path = self.dataset.iloc[idx, 0]
        gray_image = cv2.imread(gray_image_path, cv2.IMREAD_GRAYSCALE)
        hh_image = self.compute_hh(gray_image)
        if self.transform:
            hh_image = self.transform(hh_image)
        return hh_image
    
    def compute_hh(self, image):
        coeffs2 = pywt.dwt2(image, 'haar')
        LL, (LH, HL, HH) = coeffs2
        return HH
        
    def get_rgb_image(self, idx):
        rgb_image_path = self.dataset.iloc[idx, 0]
        rgb_image = Image.open(rgb_image_path)
        if self.transform:
            rgb_image = self.transform(rgb_image)
        return rgb_image
    
    def get_dct_image(self, idx):
        rgb_image_path = self.dataset.iloc[idx, 0]
        rgb_image = cv2.imread(rgb_image_path)
        dct_image = self.compute_dct_color(rgb_image)
        if self.transform:
            dct_image = self.transform(dct_image)
        
        return dct_image
    
    def get_label(self, idx):
        return self.dataset.iloc[idx, 1]
    

    def compute_dct_color(self, image):
        image_float = np.float32(image)
        dct_image = np.zeros_like(image_float)
        for i in range(3):  
            dct_image[:, :, i] = cv2.dct(image_float[:, :, i])
        return dct_image
