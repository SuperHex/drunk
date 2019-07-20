import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, dataloader
from torchvision import transforms, utils
import torch.nn as nn
import csv
from PIL import Image

COMMON_PREFIX = '/run/media/cirno/40127CD9E5B9A466/dataset/Hollywood2/'
TAG_PREFIX = os.path.join(COMMON_PREFIX, "ClipSets")
FLOW_PREFIX = os.path.join(COMMON_PREFIX, "OpticalFlows")
FLOW_TRAIN_PREFIX = os.path.join(FLOW_PREFIX, "train")
TAG_RUN_TRAIN = 'Run_train.txt'
TAG_RUN_TEST = 'Run_test.txt'
TAG_KISS_TRAIN = 'Kiss_train.txt'
TAG_KISS_TEST = 'Kiss_test.txt'

MOTION_OTHER = 0
MOTION_RUN = 1
MOTION_KISS = 2

def parse_tag(tag_abs_path):
    f = open(tag_abs_path, 'r')
    lines = [[tag for tag in line.strip('\t\n\r').split(' ') if tag != ''] for line in f.readlines()]
    f.close()
    lines = [(tag[0], int(tag[1])) for tag in lines]
    return lines

def mergeTag(tags, offset, dic):
    for tag, val in tags:
        if tag not in dic:
            tagbits = [1., 0., 0.] # three classes for now, run, kiss and other
            if val != -1:
                tagbits[MOTION_OTHER] = 0
                tagbits[offset] = 1.0
            tensor = torch.FloatTensor(np.array(tagbits))
            dic[tag] = tensor
        else:
            if val != -1:
                dic[tag][MOTION_OTHER] = 0
                dic[tag][offset] = 1.0
    return dic

def searchSegment(arr, val):
    index = 0
    while index < len(arr):
        if val < arr[index]:
            return index + 1
        index += 1
    return -1

class MotionData(Dataset):
    def __init__(self):
        self.STACK_LEN = 20
        self.WORK_DIR = FLOW_TRAIN_PREFIX
        self.FOLDER_PREFIX =  "actionclip"
        self.TRAIN_TYPE = "train"
        
        self.tag = {}
        self.stackCount = {}
        self.imgCache = {}

        self.tag = mergeTag(parse_tag(os.path.join(TAG_PREFIX, TAG_RUN_TRAIN)),
                            MOTION_RUN, self.tag)
        self.tag = mergeTag(parse_tag(os.path.join(TAG_PREFIX, TAG_KISS_TRAIN)),
                            MOTION_KISS, self.tag)

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

        # cache the total stack count
        self.length = 0
        for val in self.stackCount.values():
            self.length += val

        # pre-compute sum for quick lookup
        self.countSum = len(self.tag) * [None]
        names = list(self.tag.keys())
        names.sort()
        tmp, index = (0, 0)
        for name in names:
            tmp += self.stackCount[name]
            self.countSum[index] = tmp
            index += 1

        # transforms that used for pre-processing
        self.trans = transforms.Compose([
            transforms.Resize((224, 528)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # only one channel
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        folderNum = searchSegment(self.countSum, index)
        # folder name. e.g. actioncliptrain000001
        fmt = self.FOLDER_PREFIX + self.TRAIN_TYPE + str(folderNum).zfill(5)
        real_index = index - self.countSum[folderNum - 2]
        print(real_index)
        
        # stack[0 - 19] is x, stack [20 - 39] is y
        stack = 40 * [None]
        for idx in range(0, 20):
            img_prefix = "OF" + str(real_index + idx).zfill(4)
            img_x = os.path.join(fmt, img_prefix + "_x.jpeg")
            img_y = os.path.join(fmt, img_prefix + "_y.jpeg")
            ix = self.loadImage(os.path.join(FLOW_TRAIN_PREFIX, img_x))
            iy = self.loadImage(os.path.join(FLOW_TRAIN_PREFIX, img_y))
            stack[idx] = ix
            stack[20 + idx] = iy

        tensor = torch.cat(stack, 0)

        # Now that we have the tensor with shape (40, 224, 528),
        # we can return it as inp
        return {"image": tensor, "label": self.tag[fmt]}
        

    def numStacks(self, path):
        return self.numFiles(path) - self.STACK_LEN + 1

    def numFiles(self, path):
        _, _, files = next(os.walk(path))
        return len(files)

    def loadImage(self, path):
        # skip IO if cached
        if path in self.imgCache:
            return self.imgCache[path]
        else:
            image = Image.open(path).convert("L")
            tensor = self.trans(image)
            self.imgCache[path] = tensor
            return tensor

train_tags = mergeTag(parse_tag(os.path.join(TAG_PREFIX, TAG_RUN_TRAIN)), MOTION_RUN,
                      mergeTag(parse_tag(os.path.join(TAG_PREFIX, TAG_KISS_TRAIN)), MOTION_KISS, {}))

m = MotionData()
print(m[181]["image"].shape)
