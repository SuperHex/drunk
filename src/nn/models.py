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


class SpatialNet(nn.Module):
    def __init__(self):
        super(SpatialNet, self).__init__()
        self.features = nn.Sequential(
            # first layer
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(True),
            #nn.Conv2d(64, 64, (3, 3)),
            #nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # second layer
            nn.Conv2d(64, 128, (3, 3), padding=(1, 1)),
            nn.ReLU(True),
            #nn.Conv2d(128, 128, (3, 3)),
            #nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # third layer
            nn.Conv2d(128, 256, (3, 3), padding=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3, 3), padding=(1, 1)),
            nn.ReLU(True),
            #nn.Conv2d(256, 256, (3, 3)),
            #nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # forth layer
            nn.Conv2d(256, 512, (3, 3), padding=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(512, 512, (3, 3), padding=(1, 1)),
            nn.ReLU(True),
            #nn.Conv2d(512, 512, (3, 3)), 
            #nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            
            # fifth layer
            nn.Conv2d(512, 512, (3, 3), padding=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(512, 512, (3, 3), padding=(1, 1)),
            nn.ReLU(True),
            #nn.Conv2d(512, 512, (3, 3)),
            #nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), 
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2)
        )

    def forward(self, x):
        x = self.features(x)
        #print(x.shape)
        x = x.view(-1, 512 * 7 * 7)
        x = self.classifier(x)
        return x


class MotionNet(nn.Module):
    def __init__(self, stackLen):
        super(MotionNet, self).__init__()
        self.features = nn.Sequential(
            # first layer
            nn.Conv2d(in_channels=stackLen, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(True),
            #nn.Conv2d(64, 64, (3, 3)),
            #nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # second layer
            nn.Conv2d(64, 128, (3, 3), padding=(1, 1)),
            nn.ReLU(True),
            #nn.Conv2d(128, 128, (3, 3)),
            #nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # third layer
            nn.Conv2d(128, 256, (3, 3), padding=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3, 3), padding=(1, 1)),
            nn.ReLU(True),
            #nn.Conv2d(256, 256, (3, 3)),
            #nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # forth layer
            nn.Conv2d(256, 512, (3, 3), padding=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(512, 512, (3, 3), padding=(1, 1)),
            nn.ReLU(True),
            #nn.Conv2d(512, 512, (3, 3)), 
            #nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            
            # fifth layer
            nn.Conv2d(512, 512, (3, 3), padding=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(512, 512, (3, 3), padding=(1, 1)),
            nn.ReLU(True),
            #nn.Conv2d(512, 512, (3, 3)),
            #nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), 
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2)
        )

    def forward(self, x):
        x = self.features(x)
        #print(x.shape)
        x = x.view(-1, 512 * 7 * 7)
        x = self.classifier(x)
        return x


class ProbabilityNet(nn.Module):
    def __init__(self, spatialNetPath, motionNetPath, motionChannel):
        spatial = SpatialNet()
        spatial.load_state_dict(torch.load(spatialNetPath))
        self.spFeatures = spatial.features
        motion = MotionNet(motionChannel)
        motion.load_state_dict(torch.load(motionNetPath))
        self.moFeatures = motion.features

        self.classifier = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 3),
        )

    def forward(self, spatial, motion):
        x1 = self.spFeatures(spatial)
        x2 = self.moFeatures(motion)
        x = torch.cat((x1, x2), dim=0) # shape: (1024, 7, 7)
        x = x.view(-1, 1024 * 7 * 7)
        x = self.classifier(x)
        return x
