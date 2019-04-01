import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from PIL import Image, ImageTk
# GUI part
import tkinter as Tk
import tkinter.filedialog as TkF


def identity(x):
    return x

def softmax(x):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

# parse_table :: String -> [[String]]
def parse_table(path):
    f = open(path, 'r')
    lines = f.readlines()
    lines = [line.strip('\t\r\n').split(',') for line in lines[1:]]
    f.close()
    return lines

# get_table_data :: [[String]] -> ([String] -> Tensor) -> Map Int Tensor
def get_table_data(lines, f=identity):
    d = {}
    for line in lines:
        d[int(line[0])] = f(line)
    return d

# parse_table :: [String] -> Tensor
def parse_labels(line):
    # output binary form
    # class Tag: [hall, room, wild, street]
    # class Size: [small, medium, large]
    # class Object: [few, medium, lot]
    # All: [hall, room, wild, street, small, medium, large, few, medium, lot]
    tagDict = {"hall": [1,0,0,0], "room": [0,1,0,0],
               "wild": [0,0,1,0], "street": [0,0,0,1]}
    sizeDict = {"small": [1,0,0], "medium": [0,1,0], "large": [0,0,1]}
    objDict = {"few": [1,0,0], "medium": [0,1,0], "lot": [0,0,1]}
    # csv format: No. tag size _ _ obj ...
    d = tagDict[line[1]] + sizeDict[line[2]] + objDict[line[5]]
    return torch.FloatTensor(np.array(d))

def from_labels(label):
    # assume length(label) == 10
    t = label[0:4]
    s = label[4:7]
    o = label[7:10]
    return (t, s, o)

def get_prob(tensor):
    arr = tensor.view([10]).cpu().detach().numpy()
    (t, s, o) = from_labels(arr)
    t = softmax(t)
    s = softmax(s)
    o = softmax(o)
    tt = zip(["hall", "room", "wild", "street"], map(lambda x: round(x, 2), t))
    ss = zip(["small", "medium", "large"], map(lambda x: round(x, 2), s))
    oo = zip(["few", "medium", "lot"], map(lambda x: round(x, 2), o))
    return (tt, ss, oo)

def load_nn(net, path):
    net.load_state_dict(torch.load(path))

def save_nn(net, path):
    torch.save(net.state_dict(), path)

class BackgroundData(Dataset):
    def __init__(self, csv_path, root_dir):
        self.class_map = get_table_data(parse_table(csv_path), parse_labels)
        self.root_dir = root_dir
        self.images = []

        for i in [x for x in range(5, 24) if x != 15]:
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
        image = Image.open(sample["image"]).convert("RGB")
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
        self.fc3 = nn.Linear(114, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 34560)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        

loader = DataLoader(BackgroundData("/home/cirno/Downloads/114514 Season2/114514/123.csv", "/home/cirno/Downloads/114514 Season2/114514"), batch_size=1, shuffle=True, num_workers=4)
device = torch.device("cuda:0")
net = BGNet().to(device)

criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def train(n):
    for epoch in range(n):
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

class GUI():
    def __init__(self):
        load_nn(net, "/home/cirno/Downloads/114514 Season2/model.pkl")
        self.nn = net
        self.root = Tk.Tk()
        # draw canvas
        self.canvas = Tk.Canvas(self.root, width=400, height=320)
        self.canvas.grid(row=0, column=1)

        # draw menu
        menubar = Tk.Menu(self.root)
        filemenu = Tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.openAndDrawImage)
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)

        # draw text
        t11 = Tk.Label(self.root, text="indoor/out? ", font=("Source Han Sans CN", 15))
        t11.grid(row=1, column=0)
        t1b = Tk.Label(self.root, text="           ", font=("Source Han Sans", 15))
        t1b.grid(row=1, column=2)
        self.t1 = Tk.Label(self.root, font=("Source Han Sans CN", 13))
        self.t1.grid(row=1, column=1)
        self.t1.configure(text="N/A")

        t22 = Tk.Label(self.root, text="Scene Size: ", font=("Source Han Sans CN", 15))
        t22.grid(row=2, column=0)
        self.t2 = Tk.Label(self.root, font=("Source Han Sans CN", 13))
        self.t2.grid(row=2, column=1)
        self.t2.configure(text="N/A")

        t33 = Tk.Label(self.root, text="Object: ", font=("Source Han Sans CN", 15))
        t33.grid(row=3, column=0)
        self.t3 = Tk.Label(self.root, text="N/A", font=("Source Han Sans CN", 13))
        self.t3.grid(row=3, column=1)

        self.root.mainloop()
        
    def openAndDrawImage(self):
        filename = TkF.askopenfilename(title="Choose an image",
                                       filetypes=[("Image", ("*jpg", "*.jpeg", "*.png"))])
        img = ImageTk.PhotoImage(file=filename)
        self.canvas.create_image(0, 0, anchor=Tk.CENTER, image=img)
        self.canvas.image = img # Fuck python's GC
        # compute
        trans = transforms.Compose([
            transforms.Resize((100, 150)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image = Image.open(filename)
        tensor = trans(image).view([1,3,100,150]).to(device)
        out = self.nn(tensor)
        (t, s, o) = get_prob(out)
        self.t1.configure(text=str(list(t)))
        self.t2.configure(text=str(list(s)))
        self.t3.configure(text=str(list(o)))

if __name__ == "__main__":
    GUI()
