import gradio as gr
from PIL import Image
import numpy as np
import os
from face_cropper import detect_and_label_faces
# Define a custom function to convert an image to grayscale
def to_grayscale(input_image):
    grayscale_image = Image.fromarray(np.array(input_image).mean(axis=-1).astype(np.uint8))
    return grayscale_image


description_markdown = """
# Fake Face Detection tool made by Jaspreet Kaur , Harmanjot Kaur and Jasmine Kaur

## Usage
This tool expects a face image as input. Upon submission, it will process the image and provide an output with bounding boxes drawn on the face. Alongside the visual markers, the tool will give a detection result indicating whether the face is fake or real.

## Disclaimer
Please note that this tool is for research purposes only and may not always be 100% accurate. Users are advised to exercise discretion and supervise the tool's usage accordingly.


"""




# Create the Gradio app
app = gr.Interface(
    fn=detect_and_label_faces,
    inputs=gr.Image(type="pil"),
    outputs="image",
    # examples=[
    #     "path_to_example_image_1.jpg",
    #     "path_to_example_image_2.jpg"
    # ]
    examples=[
        os.path.join("Examples", image_name) for image_name in os.listdir("Examples")
    ],
    title="Fake Face Detection",
    description=description_markdown,
)

# Run the app
app.launch()

































# import torch.nn.functional as F
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from tqdm import tqdm
# import warnings
# warnings.filterwarnings("ignore")

# from utils.config import cfg
# from dataset.real_n_fake_dataloader import Extracted_Frames_Dataset
# from utils.data_transforms import get_transforms_train, get_transforms_val
# from net.Multimodalmodel import Image_n_DCT
# import gradio as gr




# import os
# import json
# import torch
# from torchvision import transforms
# from torch.utils.data import DataLoader, Dataset
# from PIL import Image
# import numpy as np
# import pandas as pd
# import cv2
# import argparse






# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns



    

# class Test_Dataset(Dataset):
#     def __init__(self, test_data_path = None, transform = None, image = None):
#         """
#         Args:   
#         returns:
#             """
        
#         if test_data_path is None and image is not None:
#             self.dataset = [(image, 2)]
#             self.transform = transform

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         sample_input = self.get_sample_input(idx)
#         return sample_input
    

#     def get_sample_input(self, idx):
#         rgb_image = self.get_rgb_image(self.dataset[idx][0])
#         dct_image = self.compute_dct_color(self.dataset[idx][0])
#         # label = self.get_label(idx)
#         sample_input = {"rgb_image": rgb_image, "dct_image": dct_image}

#         return sample_input
    

#     def get_rgb_image(self, rgb_image):
#         # rgb_image_path = self.dataset[idx][0]
#         # rgb_image = Image.open(rgb_image_path)
#         if self.transform:
#             rgb_image = self.transform(rgb_image)
#         return rgb_image
    
#     def get_dct_image(self, idx):
#         rgb_image_path = self.dataset[idx][0]
#         rgb_image = cv2.imread(rgb_image_path)
#         dct_image = self.compute_dct_color(rgb_image)
#         if self.transform:
#             dct_image = self.transform(dct_image)
        
#         return dct_image
    
#     def get_label(self, idx):
#         return self.dataset[idx][1]
    

#     def compute_dct_color(self, image):
#         image_float = np.float32(image)
#         dct_image = np.zeros_like(image_float)
#         for i in range(3):  
#             dct_image[:, :, i] = cv2.dct(image_float[:, :, i])
#         if self.transform:
#             dct_image = self.transform(dct_image)
#         return dct_image
    

# device = torch.device("cpu")
# # print(device)
# model = Image_n_DCT()
# model.load_state_dict(torch.load('weights/best_model.pth', map_location = device))
# model.to(device)
# model.eval()


# def classify(image):
#     test_dataset = Test_Dataset(transform = get_transforms_val(), image = image)
#     inputs = test_dataset[0]
#     rgb_image, dct_image = inputs['rgb_image'].to(device), inputs['dct_image'].to(device)
#     output = model(rgb_image.unsqueeze(0), dct_image.unsqueeze(0))
#     # _, predicted = torch.max(output.data, 1)
#     # print(f"the face is {'real' if predicted==1 else 'fake'}")
#     return {'Fake': output[0][0], 'Real': output[0][1]}

# iface = gr.Interface(fn=classify, inputs="image", outputs="label")
# if __name__ == "__main__":
#     iface.launch()
