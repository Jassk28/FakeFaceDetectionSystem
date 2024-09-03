import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

from utils.config import cfg
from dataset.real_n_fake_dataloader import Extracted_Frames_Dataset
from utils.data_transforms import get_transforms_train, get_transforms_val
from net.Multimodalmodel import Image_n_DCT



import os
import json
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import argparse

class Test_Dataset(Dataset):
    def __init__(self, test_data_path = None, transform = None, image_path = None, multi_modal = "dct"):
        """
        Args:   
        returns:
            """
        self.multi_modal = multi_modal
        if test_data_path is None and image_path is not None:
            self.dataset = [[image_path, 2]]
            self.transform = transform

        else:
            self.transform = transform
            
            self.real_data = os.listdir(test_data_path + "/real")
            self.fake_data = os.listdir(test_data_path + "/fake")
            self.dataset = []
            for image in self.real_data:
                self.dataset.append([test_data_path + "/real/" + image, 1])

            for image in self.fake_data:
                self.dataset.append([test_data_path + "/fake/" + image, 0])
                
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
        gray_image_path = self.dataset[idx][0]
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
        gray_image_path = self.dataset[idx][0]
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
        rgb_image_path = self.dataset[idx][0]
        rgb_image = Image.open(rgb_image_path)
        if self.transform:
            rgb_image = self.transform(rgb_image)
        return rgb_image
    
    def get_dct_image(self, idx):
        rgb_image_path = self.dataset[idx][0]
        rgb_image = cv2.imread(rgb_image_path)
        dct_image = self.compute_dct_color(rgb_image)
        if self.transform:
            dct_image = self.transform(dct_image)
        
        return dct_image
    
    def get_label(self, idx):
        return self.dataset[idx][1]
    

    def compute_dct_color(self, image):
        image_float = np.float32(image)
        dct_image = np.zeros_like(image_float)
        for i in range(3):  
            dct_image[:, :, i] = cv2.dct(image_float[:, :, i])
        return dct_image

    
class Test:
    def __init__(self, model_paths = [ 'weights/faceswap-hh-best_model.pth',
                                      'weights/faceswap-fft-best_model.pth',
                                                                            ],
                 multi_modal = ["hh","fct"]):
        self.model_path = model_paths
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        # Load the model
        self.model1 = Image_n_DCT()
        self.model1.load_state_dict(torch.load(self.model_path[0], map_location = self.device))
        self.model1.to(self.device)
        self.model1.eval()

        self.model2 = Image_n_DCT()
        self.model2.load_state_dict(torch.load(self.model_path[1], map_location = self.device))
        self.model2.to(self.device)
        self.model2.eval()


        self.multi_modal = multi_modal


    def testimage(self, image_path):
        test_dataset1 = Test_Dataset(transform = get_transforms_val(), image_path = image_path, multi_modal = self.multi_modal[0])
        test_dataset2 = Test_Dataset(transform = get_transforms_val(), image_path = image_path, multi_modal = self.multi_modal[1])

        inputs1 = test_dataset1[0]
        rgb_image1, dct_image1 = inputs1['rgb_image'].to(self.device), inputs1['dct_image'].to(self.device)

        inputs2 = test_dataset2[0]
        rgb_image2, dct_image2 = inputs2['rgb_image'].to(self.device), inputs2['dct_image'].to(self.device)

        output1 = self.model1(rgb_image1.unsqueeze(0), dct_image1.unsqueeze(0))

        output2 = self.model2(rgb_image2.unsqueeze(0), dct_image2.unsqueeze(0))

        output = (output1 + output2)/2
        # print(output.shape)
        _, predicted = torch.max(output.data, 1)
        return 'real' if predicted==1 else 'fake'