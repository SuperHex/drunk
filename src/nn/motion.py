#! /usr/bin/python3

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
import random
from PIL import Image


COMMON_PREFIX = '/run/media/cirno/40127CD9E5B9A466/dataset/shabixi/'
TAG_PREFIX = os.path.join(COMMON_PREFIX, "ClipSets")
FLOW_PREFIX = os.path.join(COMMON_PREFIX, "OpticalFlows")
FLOW_TRAIN_PREFIX = os.path.join(FLOW_PREFIX, "train_small")
TAG_RUN_TRAIN = 'Run_train.txt'
TAG_RUN_TEST = 'Run_test.txt'
TAG_KISS_TRAIN = 'Kiss_train.txt'
TAG_KISS_TEST = 'Kiss_test.txt'

MOTION_OTHER = 0
MOTION_RUN = 1
MOTION_KISS = 2
MOTION_DRIVE = 3
MOTION_FIGHT = 4
MOTION_HANDSHAKE = 5

def parse_tag(tag_abs_path):
    f = open(tag_abs_path, 'r')
    lines = [[tag for tag in line.strip('\t\n\r').split(' ') if tag != ''] for line in f.readlines()]
    f.close()
    lines = [(tag[0], int(tag[1])) for tag in lines]
    return lines

def mergeTagBCE(tags, offset, dic):
    for tag, val in tags:
        if tag not in dic:
            tagbits = [1, 0, 0] # three classes for now, run, kiss and other
            if val != -1:
                tagbits[MOTION_OTHER] = 0
                tagbits[offset] = 1
            tensor = torch.FloatTensor(np.array(tagbits))
            dic[tag] = tensor
        else:
            if val != -1:
                dic[tag][MOTION_OTHER] = 0
                dic[tag][offset] = 1
    return dic

def mergeTagCross(tags, offset, dic):
    for tag, val in tags:
        if tag not in dic:
            tagbits = 1  # 1: other, offset: class
            if val != -1:
                tagbits = offset
            #tensor = torch.LongTensor(np.array([tagbits]))
            dic[tag] = tagbits
        else:
            if val != -1:
                dic[tag] = offset # torch.LongTensor(np.array([offset]))
    return dic

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

imgTrans = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # only one channel
])

class MotionData(Dataset):
    def __init__(self, ttype="train", dirname="train"):
        self.STACK_LEN = 20
        self.WORK_DIR = os.path.join(FLOW_PREFIX, dirname)
        self.FOLDER_PREFIX =  "actionclip"
        self.TRAIN_TYPE = ttype
        
        self.tag = {}
        self.stackCount = {}
        self.imgCache = {}

        # if ttype == "train":
        #     tag_run = "Run_train.txt"
        #     tag_kiss = "Kiss_train.txt"
        #     tag_drive = "DriveCar_train.txt"
        #     tag_fight = "FightPerson_train.txt"
        #     tag_handshake = "HandShake_train.txt"
        # else:
        #     tag_run = "Run_test.txt"
        #     tag_kiss = "Kiss_test.txt"
        #     tag_drive = "DriveCar_test.txt"
        #     tag_fight = "FightPerson_test.txt"
        #     tag_handshake = "HandShake_test.txt"
        
        # self.tag = mergeTagCross(parse_tag(os.path.join(TAG_PREFIX, tag_run)),
        #                     MOTION_RUN, self.tag)
        # self.tag = mergeTagCross(parse_tag(os.path.join(TAG_PREFIX, tag_kiss)),
        #                     MOTION_KISS, self.tag)
        # self.tag = mergeTagCross(parse_tag(os.path.join(TAG_PREFIX, tag_drive)), MOTION_DRIVE, self.tag)
        # self.tag = mergeTagCross(parse_tag(os.path.join(TAG_PREFIX, tag_fight)), MOTION_FIGHT, self.tag)
        # self.tag = mergeTagCross(parse_tag(os.path.join(TAG_PREFIX, tag_handshake)), MOTION_HANDSHAKE, self.tag)
        self.tag = mergeTagCross(parse_tag(os.path.join(TAG_PREFIX, "Walk_train.txt")), 2, self.tag)
        print(self.tag)

        # get file count from cache. If not found, create one
        cacheFile = os.path.join(self.WORK_DIR, ".stackcount")
        if os.path.isfile(cacheFile):
            r = csv.reader(open(cacheFile, "r"))
            for row in r:
                name, count = row
                if name in self.stackCount.keys():
                    print("WARNING: name collision in cache file")
                self.stackCount[name] = int(count)
        else:
            self.stackCount = dict([(dir, self.numStacks(
                os.path.join(self.WORK_DIR, dir)))
                    for dir in os.listdir(self.WORK_DIR)])
            w = csv.writer(open(cacheFile, "w"))
            items = list(self.stackCount.items())
            items.sort()
            for key, val in items:
                w.writerow([key, val])

        # pre-compute sum for quick lookup
        self.countSum = len(self.tag) * [None]
        names = list(self.tag.keys())
        names.sort()
        tmp, index = (0, 0)
        for name in names:
            if name in self.stackCount:
                tmp += self.stackCount[name]
            self.countSum[index] = tmp
            index += 1

        self.length = self.countSum[-1]

        # transforms that used for pre-processing
        self.trans = imgTrans

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        folderNum, real_index = searchSegment(self.countSum, index)
        # folder name. e.g. actioncliptrain000001
        #fmt = self.FOLDER_PREFIX + self.TRAIN_TYPE + str(folderNum).zfill(5)
        fmt = str(folderNum).zfill(2)
        
        # stack[0 - 9] is x, stack [10 - 19] is y
        stack = self.STACK_LEN * 2 * [None]
        for idx in range(0, self.STACK_LEN):
            img_prefix = "OF" + str(real_index + idx).zfill(4)
            img_x = os.path.join(fmt, img_prefix + "_x.jpeg")
            img_y = os.path.join(fmt, img_prefix + "_y.jpeg")
            ix = self.loadImage(os.path.join(self.WORK_DIR, img_x))
            iy = self.loadImage(os.path.join(self.WORK_DIR, img_y))
            stack[idx] = ix
            stack[self.STACK_LEN + idx] = iy

        tensor = torch.cat(stack, 0)

        # Now that we have the tensor with shape (40, 224, 528),
        # we can return it as inp
        #print("tag for {0} is {1}".format(index, self.tag[fmt]))
        return {"image": tensor, "label": self.tag[fmt]}
        

    def numStacks(self, path):
        # divide by 2 because of (x, y)
        return self.numFiles(path) // 2 - self.STACK_LEN + 1

    def numFiles(self, path):
        _, _, files = next(os.walk(path))
        return len(files)

    def loadImage(self, path):
        # skip IO if cached
        # if path in self.imgCache:
        #     return self.imgCache[path]
        # else:
            image = Image.open(path).convert("L")
            tensor = self.trans(image)
            #self.imgCache[path] = tensor
            return tensor

class MotionNet(nn.Module):
    def __init__(self):
        super(MotionNet, self).__init__()
        self.conv1 = nn.Conv2d(40, 60, 6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(60, 80, 6)
        self.conv3 = nn.Conv2d(80, 120, 6)
        self.conv4 = nn.Conv2d(120, 140, 6)
        # final output shape (200, 9, 28)
        self.fc1 = nn.Linear(140 * 2 * 2, 2500)
        self.fc3 = nn.Linear(2500, 1250)
        self.fc4 = nn.Linear(1250, 320)
        self.fc5 = nn.Linear(320, 72)
        self.fc6 = nn.Linear(72, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x), True))
        x = self.pool(F.relu(self.conv2(x), True))
        x = self.pool(F.relu(self.conv3(x), True))
        x = self.pool(F.relu(self.conv4(x), True))
        x = x.view(-1, 140 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        return x

def train(n):
    print("start training!")
    print("Dataset size: {0}".format(m.length))

    for epoch in range(n):
        run_loss = 0.0

        for i, data in enumerate(loader, 0):
            stream = data["image"].cuda()
            label = data["label"].cuda()

            optimizer.zero_grad()

            output = net(stream)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            run_loss += float(loss.item())

            if i % 10 == 9:
                print('[%d, %6d] loss = %.3f' %
                    (epoch + 1, i + 1, run_loss / 10))
                run_loss = 0.0
    print("Done!")

if __name__ == "__main__":
    argv = sys.argv

    if argv[1] == "train":
        net = MotionNet().cuda()
        m = MotionData(dirname="train")
        loader = DataLoader(m, batch_size=32, shuffle=True, num_workers=0)

        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam(net.parameters())

        if os.path.isfile(os.path.join(COMMON_PREFIX, "model.pkl")):
            net.load_state_dict(torch.load(os.path.join(COMMON_PREFIX, "model.pkl")))

        for i in range(5):
            train(2)
            torch.save(net.state_dict(), os.path.join(COMMON_PREFIX, "model.pkl"))
            print("model saved to {0}".format(COMMON_PREFIX + "model.pkl"))
    else:
        net = MotionNet().cpu()
        net.load_state_dict(torch.load(os.path.join(COMMON_PREFIX, "model.pkl")))
        net.eval()
        loader = MotionData("test", "test")
        ran = random.randrange(0, len(loader))
        l = loader[ran]
        print("Select data {0}".format(ran))
        infer = net(l["image"].unsqueeze(0)).view([3]).cpu().detach()
        print(infer)
        print("Infered tag: {0}, actual tag: {1}".format(torch.max(infer, 0)[1], l["label"]))
