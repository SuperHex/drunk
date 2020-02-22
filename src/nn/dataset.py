from label import buildLabelMap
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
from PIL import Image
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def searchSegment(arr, val):
    index = 0
    while index < len(arr):
        if val < arr[index]:
            if index is 0:
                return index + 1, val
            else:
                return index + 1, val - arr[index - 1]
        index += 1
    return None, None

# Load RGB images
class SpatialData(Dataset):
    def __init__(self, folderPath, labelPath):
        self.folderPath = folderPath

        # get all directories
        directories = next(os.walk(folderPath))[1]
        directories.sort(key=natural_keys)

        self.countSum = [None] * len(directories)
        total = 0
        # compute number of images
        for directory in directories:
            absPath = os.path.join(folderPath, directory)
            folderIndex = int(directory)
            nbFiles = self.numFiles(absPath)
            total += nbFiles
            self.countSum[folderIndex - 1] = total
        
        print(self.countSum)
        self.length = self.countSum[-1]

        # build labels
        self.labels = buildLabelMap(labelPath)
        #print(self.labels)

        # transforms that used for pre-processing
        self.trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        folderNum, real_index = searchSegment(self.countSum, index)
        # folder name. e.g. actioncliptrain000001
        #fmt = self.FOLDER_PREFIX + self.TRAIN_TYPE + str(folderNum).zfill(5)
        fmt = str(folderNum).zfill(2)
        
        img_prefix = "_RGB_" + str(real_index).zfill(4)
        imgName = fmt + img_prefix + ".jpeg"
    
        tensor = self.loadImage(os.path.join(os.path.join(self.folderPath, fmt), imgName))

        tag = 0
        if real_index in self.labels[folderNum]:
            tag = 1
        
        return {"image": tensor, "label": tag}

    def numFiles(self, path):
        print(path)
        _, _, files = next(os.walk(path))
        return len(files)

    def loadImage(self, path):
        # skip IO if cached
        # if path in self.imgCache:
        #     return self.imgCache[path]
        # else:
            image = Image.open(path).convert("RGB")
            tensor = self.trans(image)
            #self.imgCache[path] = tensor
            return tensor