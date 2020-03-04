from label import buildLabelMap, buildLabelBCE
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
    def __init__(self, folderPath, labelPath=None, videoName=None):
        self.folderPath = folderPath
        self.folders = []

        if videoName is None:
            # get all directories
            directories = next(os.walk(folderPath))[1]
            directories.sort(key=natural_keys)

            self.countSum = []
            self.folders = []
            total = 0
            # compute number of images
            for directory in directories:
                absPath = os.path.join(folderPath, directory)
                folderIndex = int(directory)
                self.folders.append(folderIndex)
                nbFiles = self.numFiles(absPath)
                total += nbFiles
                self.countSum.append(total)
            
            #print(self.countSum)
            self.length = self.countSum[-1]
        else:
            self.folders.append(int(videoName))
            self.countSum = self.numFiles(os.path.join(folderPath, videoName))
            self.length = self.countSum[0]

        # build labels
        if labelPath is None:
            self.labels = None
        else:
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
        folderIndex, real_index = searchSegment(self.countSum, index)
        folderNum = self.folders[folderIndex - 1]  # folderIndex is 1 based
        # folder name. e.g. actioncliptrain000001
        #fmt = self.FOLDER_PREFIX + self.TRAIN_TYPE + str(folderNum).zfill(5)
        fmt = str(folderNum).zfill(2)
        
        img_prefix = "_RGB_" + str(real_index).zfill(4)
        imgName = fmt + img_prefix + ".jpeg"
    
        tensor = self.loadImage(os.path.join(os.path.join(self.folderPath, fmt), imgName))

        tag = 0
        if self.labels is not None and real_index in self.labels[folderNum]:
            tag = 1
        
        return {"image": tensor, "label": tag}

    def numFiles(self, path):
        print(path)
        _, _, files = next(os.walk(path))
        return len(files)

    def loadImage(self, path):
        image = Image.open(path).convert("RGB")
        tensor = self.trans(image)
        return tensor

# Optical Flow dataset loader
class MotionData(Dataset):
    def __init__(self, stackLen, folderPath, labelPath=None, videoName=None):
        self.STACK_LEN = stackLen
        self.OFFSET = self.STACK_LEN // 2
        self.first_layer_channels = self.STACK_LEN * 2

        self.folderPath = folderPath
        self.folders = []
        
        self.stackCount = {}

        self.labels = None if labelPath is None else buildLabelMap(labelPath)

        # get file count from cache. If not found, create one
        cacheFile = os.path.join(self.folderPath, ".stackcount")
        if os.path.isfile(cacheFile):
            r = csv.reader(open(cacheFile, "r"))
            for row in r:
                if not row:
                    continue
                name, count = row
                if name in self.stackCount.keys():
                    print("WARNING: name collision in cache file")
                self.stackCount[int(name)] = int(count)
        else:
            _, directories, _ = next(os.walk(self.folderPath))
            directories.sort()
            self.stackCount = dict([(int(dir), self.numStacks(
                os.path.join(self.folderPath, dir))) for dir in directories])
            w = csv.writer(open(cacheFile, "w"))
            items = list(self.stackCount.items())
            items.sort()
            for key, val in items:
                w.writerow([key, val])

        # pre-compute sum for quick lookup
        self.countSum = []
        names = list(self.stackCount.keys())
        names.sort()
        self.folders = names

        if videoName is None:
            tmp = 0
            for name in names:
                if name in self.stackCount:
                    tmp += self.stackCount[name]
                self.countSum.append(tmp)

            #print(self.countSum)
            self.length = self.countSum[-1]

        else:
            video = int(videoName)
            if video in set(names):
                self.countSum.append(self.stackCount[video])
            else:
                print("Video " + videoName + " not found!")
                exit(1)
            self.length = self.countSum[0]

        # transforms that used for pre-processing
        self.trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Index is OF index

        folderIndex, of_index = searchSegment(self.countSum, index)
        center_index = of_index + self.OFFSET
        folderNum = self.folders[folderIndex - 1]  # folderIndex is 1 based
        fmt = str(folderNum).zfill(2)
        
        # stack[0 - 9] is x, stack [10 - 19] is y
        stack = self.STACK_LEN * 2 * [None]
        for idx in range(-1 * self.OFFSET, self.STACK_LEN):
            img_prefix = "OF" + str(center_index + idx).zfill(4)
            img_x = os.path.join(fmt, img_prefix + "_x.jpeg")
            img_y = os.path.join(fmt, img_prefix + "_y.jpeg")
            ix = self.loadImage(os.path.join(self.folderPath, img_x))
            iy = self.loadImage(os.path.join(self.folderPath, img_y))
            stack[idx] = ix
            stack[self.STACK_LEN + idx] = iy

        tensor = torch.cat(stack, 0)

        if self.labels is None:
            return {'image': tensor, 'label': 0}

        tags = []
        for i in range(-1 * self.OFFSET, self.STACK_LEN):
            if i in self.labels[folderNum]:
                tags.append(True)
            else:
                tags.append(False)
        
        tag = 1 if all(tags[:self.OFFSET + 1]) or all(tags[self.OFFSET:]) else 0
        
        return {"image": tensor, "label": tag}
        

    def numStacks(self, path):
        # divide by 2 because of (x, y)
        return self.numFiles(path) // 2 - self.STACK_LEN + 1

    def numFiles(self, path):
        print(path)
        _, _, files = next(os.walk(path))
        return len(files)

    def loadImage(self, path):
        image = Image.open(path).convert("L")
        tensor = self.trans(image)
        return tensor

class ProbabilityData(Dataset):
    def __init__(self, spatialFolder, MotionFolder, labelPath=None, videoName=None):
        self.spatial = SpatialData(spatialFolder, labelPath, videoName)
        self.motion = MotionData(MotionFolder, labelPath, videoName)

        self.labels = buildLabelBCE(labelPath) if labelPath is not None else None

    def __len__(self):
        return self.motion.__len__()

    def __getitem__(self, index):
        x = self.spatial.__getitem__(index)
        y = self.motion.__getitem__(index)

        image = (x['image'], y['image'])
        actioness, start, end = 0, 0, 0

        if self.labels is not None:
            if index in self.labels['actioness']:
                actioness = 1

            if index in self.labels['start']:
                start = 1

            if index in self.labels['end']:
                end = 1

        return {'image': image, 'label': torch.Tensor([actioness, start, end])}