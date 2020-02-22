#! /usr/local/bin/python3

import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import random
from PIL import Image
import optical_flow as of
import re
import cv2 as cv

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MOTION_OTHER = 0
MOTION_WALK = 1

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def parse_tag(tag_abs_path):
    f = open(tag_abs_path, 'r')
    lines = [[tag for tag in line.strip('\t\n\r').split(' ') if tag != ''] for line in f.readlines()]
    f.close()
    #lines = [(tag[0], int(tag[1])) for tag in lines]
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

def mergeTagCross_(tags, dic):
    for folder, frame in tags:
        if folder not in dic:
            dic[folder] = set([frame])
        else:
            dic[folder].add(frame) # torch.LongTensor(np.array([offset]))
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
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # only one channel
])

class MotionData(Dataset):
    def __init__(self, ttype="train", dirname="train"):
        self.STACK_LEN = 1
        self.WORK_DIR = os.path.join(FLOW_PREFIX, dirname)
        self.FOLDER_PREFIX =  "actionclip"
        self.TRAIN_TYPE = ttype
        
        self.tag = {}
        self.stackCount = {}
        self.imgCache = {}
        self.videos = {}
        self.length = 0

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
        self.tag = mergeTagCross_(parse_tag(os.path.join(TAG_PREFIX, "1.txt")), self.tag)
        print(self.tag)
        # get file count from cache. If not found, create one
        # pre-compute sum for quick lookup
        self.countSum = []
        files = [f for f in os.listdir(self.WORK_DIR) if os.path.isfile(os.path.join(self.WORK_DIR, f))]
        files.sort(key=natural_keys)

        tmp = 0
        for f in files:
            cap = cv.VideoCapture(os.path.join(self.WORK_DIR, f))
            caplen = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            print(f, caplen)
            self.videos[os.path.splitext(f)[0]] = cap
            tmp += caplen - self.STACK_LEN + 1
            self.countSum.append(tmp)
        print(self.countSum)
        self.length = self.countSum[-1]

        # transforms that used for pre-processing
        self.trans = imgTrans

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        folderNum, real_index = searchSegment(self.countSum, index)
        # folder name. e.g. actioncliptrain000001
        #fmt = self.FOLDER_PREFIX + self.TRAIN_TYPE + str(folderNum).zfill(5)
        ##fmt = str(folderNum).zfill(2)
        fmt = str(folderNum)  # for small tests
        
        # stack[0 - 9] is x, stack [10 - 19] is y
        stack = self.STACK_LEN * [None]
        for idx in range(0, self.STACK_LEN):
            self.videos[str(folderNum)].set(cv.CAP_PROP_POS_FRAMES, real_index + idx)
            arr = self.videos[str(folderNum)].read()[1]
            im = Image.fromarray(arr)
            stack[idx] = self.trans(im)

        tensor = torch.cat(stack, 0)

        # Now that we have the tensor with shape (40, 224, 528),
        # we can return it as inp
        #print("tag for {0} is {1}".format(index, self.tag[fmt]))
        tag = 1 if real_index in self.tag[fmt] or real_index + 1 in self.tag[fmt] or real_index + 2 in self.tag[fmt] else 0
        return {"image": tensor, "label": tag}
        

    def numStacks(self, path):
        # divide by 2 because of (x, y)
        return self.numFiles(path) // 2 - self.STACK_LEN + 1

    def numFiles(self, path):
        print(path)
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
        self.conv1 = nn.Conv2d(3, 60, 6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(60, 80, 6)
        self.conv3 = nn.Conv2d(80, 120, 6)
        self.conv4 = nn.Conv2d(120, 140, 6)
        # final output shape (200, 9, 28)
        self.fc1 = nn.Linear(140 * 2 * 2, 2500)
        self.fc3 = nn.Linear(2500, 1250)
        self.fc4 = nn.Linear(1250, 320)
        self.fc5 = nn.Linear(320, 72)
        self.fc6 = nn.Linear(72, 2)

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
    for epoch in range(n):
        run_loss = 0.0

        for i, data in enumerate(loader, 0):
            stream = data["image"].to(device)
            label = data["label"].to(device)

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

    if len(argv) < 3:
        print("Usage: motion.py [train/test] [dir of Clipsets]")

    COMMON_PREFIX = argv[2]
    TAG_PREFIX = os.path.join(COMMON_PREFIX, "ClipSets")
    FLOW_PREFIX = os.path.join(COMMON_PREFIX, "OpticalFlows")
    FLOW_TRAIN_PREFIX = os.path.join(FLOW_PREFIX, "train_small")

    if argv[1] == "train":
        net = MotionNet().to(device)
        m = MotionData(dirname="train")
        loader = DataLoader(m, batch_size=32, shuffle=True, num_workers=0)

        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.1, 1.84])).float()).to(device)
        optimizer = optim.Adam(net.parameters())

        if os.path.isfile(os.path.join(COMMON_PREFIX, "model.pkl")):
            net.load_state_dict(torch.load(os.path.join(COMMON_PREFIX, "model.pkl")))

        print("start training!")
        print("Dataset size: {0}".format(m.length))
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
