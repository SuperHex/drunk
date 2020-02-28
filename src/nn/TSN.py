from label import buildLabelMap
from dataset import SpatialData, MotionData
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
import random
import re
import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

def plot_loss(fig):
    plt.ion()
    ax = fig.add_subplot(111)
    ln, = plt.plot([], [], 'ro')
    plt.title('Loss')
    ax.set_autoscale_on(True) # enable autoscale
    ax.autoscale_view(True,True,True)
    return ax, ln,

def update_loss(ax, line, x, y):
    line.set_data(y, x)
    ax.relim()
    plt.draw()

def train(epochs, path, save_per_epoch=True):
    print("start training!")
    print("Dataset size: {0}".format(m.length))

    xdata = []
    ydata = []
    for epoch in range(epochs):
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

            # draw loss
            xdata.append(epoch * m.length + i)
            ydata.append(float(loss.item()))
            update_loss(ax, line, xdata, ydata)

            if i % 10 == 9:
                print('[%d, %6d] loss = %.5f' %
                    (epoch + 1, i + 1, run_loss / 10))
                run_loss = 0.0

        if save_per_epoch:
            torch.save(net.state_dict(), os.path.join(path, "model.pkl"))
            print("model saved to {0}".format(folder + "model.pkl"))
    print("Done!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train/Test the two stream network")
    parser.add_argument('--action', dest='action', default='train', help='Options: train | test. default: [train]')
    parser.add_argument('--net', dest='net', help='Options: spatial | motion')
    parser.add_argument('--path', dest='folder', help='Dataset folder')
    parser.add_argument('--tag', dest='label', help='Tag file path')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-3, help='Learning rate. default: [1e-3]')
    parser.add_argument('--epochs', dest='epochs', type=int, default=10, help='Total training rounds. default: [10]')
    parser.add_argument('--batch', dest='batch', type=int, default=32, help='Batch size in training. default: [32]')
    parser.add_argument('--no-save-per-epoch', dest="no_save_per_epoch", default=False, action='store_true', help="Don't save trained model per epoch")

    args = parser.parse_args()

    print(args.learning_rate)

    folder = args.folder
    label = args.label

    if args.action == 'train':

        net = None
        m = None
        if args.net == 'spatial':
            net = SpatialNet().to(device)
            m = SpatialData(folder, label)
        elif args.net == 'motion':
            m = MotionData(folder, label)
            net = MotionNet(m.first_layer_channels).to(device)
            
        loader = DataLoader(m, batch_size=args.batch, shuffle=True, num_workers=0)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

        if os.path.isfile(os.path.join(folder, "model.pkl")):
            net.load_state_dict(torch.load(os.path.join(folder, "model.pkl")))

        fig = plt.figure()
        ax, line = plot_loss(fig)
        train(args.epochs, folder, not args.no_save_per_epoch)

        plt.ioff()
        plt.show()

    elif args.action == 'test':
        net = SpatialNet().cpu()
        net.load_state_dict(torch.load(os.path.join(folder, "model.pkl")))
        net.eval()
        loader = SpatialData(folder, label)
        ran = random.randrange(0, len(loader))
        l = loader[ran]
        print("Select data {0}".format(ran))
        infer = net(l["image"].unsqueeze(0)).view([2]).cpu().detach()
        print(infer)
        print("Infered tag: {0}, actual tag: {1}".format(torch.max(infer, 0)[1], l["label"]))