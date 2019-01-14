import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image


def identity(x):
    return x

def parse_table(path):
    f = open(path, 'r')
    lines = f.readlines()
    lines = [line.strip('\t\r\n').split(',') for line in lines[1:]]
    f.close()
    return lines

def get_table_data(lines, f=identity):
    d = {}
    for line in lines:
        d[int(line[0])] = f(line)
    return d

class BackgroundData(Dataset):
    def __init__(self, csv_path, root_dir):
        dic = {"small": 0, "medium": 1, "large": 2}
        self.class_map = get_table_data(parse_table(csv_path), lambda x : dic[x[2]])
        self.root_dir = root_dir
        self.images = []

        for i in range(5, 14):
            folder_path = os.path.join(self.root_dir, str(i))
            images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            images = map(
                lambda img: {"image": img, "label": self.class_map[i]}, images)
            self.images += images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = self.images[index]
        trans = transforms.Compose([
            transforms.Resize((100, 150)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image = Image.open(sample["image"])
        samp = {"image": trans(image), "label": sample["label"]}
        return samp

class BGNet(nn.Module):
    def __init__(self):
        super(BGNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 64, 5)
        self.conv3 = nn.Conv2d(64, 256, 5)
        self.fc1 = nn.Linear(34560, 514)
        self.fc2 = nn.Linear(514, 114)
        self.fc3 = nn.Linear(114, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 34560)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
        

loader = DataLoader(BackgroundData("/home/cirno/Downloads/114514/data.csv", "/home/cirno/Downloads/114514"), batch_size=1, shuffle=True, num_workers=4)
device = torch.device("cuda:0")
net = BGNet().to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def train():
    for epoch in range(100):
        run_loss = 0.0

        for i, data in enumerate(loader, 0):

            inputs = data["image"].to(device)
            labels = torch.tensor(data["label"], device=device)
            
            optimizer.zero_grad()
                
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            run_loss += loss.item()
            
            if i % 20 == 19:    # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, run_loss / 20))
                run_loss = 0.0
    print("Done!")


trans = transforms.Compose([
    transforms.Resize((100, 150)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
