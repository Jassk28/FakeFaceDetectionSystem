from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C

# 0. basic config
__C.TAG = 'default'
__C.CLASSES = ['Real', 'Fake']


# config of network input
__C.MULTIMODAL_FUSION = edict()
__C.MULTIMODAL_FUSION.IMG_CHANNELS = [3, 64, 128, 256, 512]
__C.MULTIMODAL_FUSION.DCT_CHANNELS = [1, 64, 128, 256, 512]


__C.NUM_EPOCHS = 100

__C.BATCH_SIZE = 64

__C.NUM_WORKERS = 4

__C.LEARNING_RATE = 0.0001

__C.PRETRAINED = False

__C.PRETRAINED_PATH = "/home/user/Documents/Real_and_DeepFake/src/best_model.pth"




__C.TEST_BATCH_SIZE = 512

__C.TEST_CSV = "/home/user/Documents/Real_and_DeepFake/src/dataset/extended_val.csv"

__C.MODEL_PATH = "/home/user/Documents/Real_and_DeepFake/src/best_model.pth"

