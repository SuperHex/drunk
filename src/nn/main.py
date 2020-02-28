from label import buildLabelMap
from dataset import SpatialData, MotionData, ProbabilityData
from models import SpatialNet, MotionNet, ProbabilityNet
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
            #xdata.append(epoch * m.length + i)
            #ydata.append(float(loss.item()))
            #update_loss(ax, line, xdata, ydata)

            if i % 10 == 9:
                print('[%d, %6d] loss = %.5f' %
                    (epoch + 1, i + 1, run_loss / 10))
                run_loss = 0.0

        if save_per_epoch:
            torch.save(net.state_dict(), path)
            print("model saved to {0}".format(folder + "model.pkl"))
    print("Done!")

def train_prob(epochs, path, save_per_epoch=True):
    print("start training!")
    print("Dataset size: {0}".format(len(m)))

    for epoch in range(epochs):
        run_loss = 0.0

        for i, data in enumerate(loader, 0):
            img1, img2 = data["image"]
            label = data["label"].to(device)

            optimizer.zero_grad()

            output = net(img1.to(device), img2.to(device))
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            run_loss += float(loss.item())

            if i % 10 == 9:
                print('[%d, %6d] loss = %.5f' %
                    (epoch + 1, i + 1, run_loss / 10))
                run_loss = 0.0

        if save_per_epoch:
            torch.save(net.state_dict(), path)
            print("model saved to {0}".format(path))
    print("Done!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train/Test the two stream network")
    parser.add_argument('--action', dest='action', default='train', help='Options: train | test. default: [train]')
    parser.add_argument('--net', dest='net', help='Options: spatial | motion | prob')
    parser.add_argument('--path', dest='folder', nargs='+', help='Dataset folder. Can enter multiple folders. e.g. `--path /path/image /path/optical_flow`')
    parser.add_argument('--tag', dest='label', default='', help='Tag file path')
    parser.add_argument('--models', nargs='*', help='Pretrained models. Can be multiple `--models /foo/model.pkl /bar/model.pkl')
    parser.add_argument('--output', dest='output_path', help="Path to save/load the model (include model name)")
    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-3, help='Learning rate. default: [1e-3]')
    parser.add_argument('--epochs', dest='epochs', type=int, default=10, help='Total training rounds. default: [10]')
    parser.add_argument('--batch', dest='batch', type=int, default=32, help='Batch size in training. default: [32]')
    parser.add_argument('--weight', type=float, nargs='*', help='Weights')
    parser.add_argument('--no-save-per-epoch', dest="no_save_per_epoch", default=False, action='store_true', help="Don't save trained model per epoch")

    args = parser.parse_args()

    folder = args.folder
    label = args.label
    output_path = args.output_path
    weight = torch.Tensor(args.weight).to(device)
    models = args.models

    if args.action == 'train':

        if args.net == 'spatial':
            net = SpatialNet().to(device)
            m = SpatialData(folder[0], label)

            if os.path.isfile(output_path):
                net.load_state_dict(torch.load(output_path))
                print('Loaded pretrained model: ' + output_path)

            loader = DataLoader(m, batch_size=args.batch, shuffle=True, num_workers=0)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

            train(args.epochs, output_path, not args.no_save_per_epoch)
        elif args.net == 'motion':
            m = MotionData(folder[0], label)
            net = MotionNet(m.first_layer_channels).to(device)

            if os.path.isfile(output_path):
                net.load_state_dict(torch.load(output_path))
                print('Loaded pretrained model: ' + output_path)

            loader = DataLoader(m, batch_size=args.batch, shuffle=True, num_workers=0)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

            train(args.epochs, output_path, not args.no_save_per_epoch)
        elif args.net == 'prob':
            m = ProbabilityData(folder[0], folder[1], label)
            net = ProbabilityNet(models[0], models[1], m.motion.first_layer_channels).to(device)

            for param in net.spFeatures.parameters():
                param.requires_grad = False
            for param in net.moFeatures.parameters():
                param.requires_grad = False

            loader = DataLoader(m, batch_size=args.batch, shuffle=True, num_workers=0)
            criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
            optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

            train_prob(args.epochs, output_path)

        #fig = plt.figure()
        #ax, line = plot_loss(fig)
        

        #plt.ioff()
        #plt.show()

    elif args.action == 'test':

        if args.net == 'prob':
            print('Infering for prob net...')
            net = ProbabilityNet(models[0], models[1], 5).cpu()
            net.load_state_dict(torch.load(output_path))
            net.eval()
            loader = ProbabilityData(folder[0], folder[1], label)

            action, start, end = [], [], []
            xdata = []
            for i in range(len(loader)):
                data = loader[i]
                infer = torch.sigmoid(net(data['image'].unsqueeze(0)).view(3).cpu().detach())
                inferl = infer.tolist()
                xdata.append(i)
                action.append(inferl[0])
                start.append(inferl[1])
                end.append(inferl[2])

            fig = plt.figure()
            ax = fig.add_subplot(311)
            ax.plot(xdata, action)
            
            sx = fig.add_subplot(312)
            sx.plot(xdata, start)

            ex = fig.add_subplot(313)
            ex.plot(xdata, end)

            plt.show()

        else:
            net = SpatialNet().cpu()
            net.load_state_dict(torch.load(output_path))
            net.eval()
            loader = SpatialData(folder[0], label)
            ran = random.randrange(0, len(loader))
            l = loader[ran]
            print("Select data {0}".format(ran))
            infer = net(l["image"].unsqueeze(0)).view([2]).cpu().detach()
            print(infer)
            print("Infered tag: {0}, actual tag: {1}".format(torch.max(infer, 0)[1], l["label"]))