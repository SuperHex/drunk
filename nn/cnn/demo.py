import optical_flow as of
import motion
import cv2 as cv
import torch
from PIL import Image
import os


class OFloader:
    def __init__(self, path):
        self.stackLen = 20
        self.video = cv.VideoCapture(path)
        self.xbuffer = [None] * self.stackLen
        self.ybuffer = [None] * self.stackLen

    def __iter__(self):
        self.f1 = of.sampleFrame(self.video)
        if self.f1 is None:
            raise StopIteration
        return self

    def __next__(self):
        for i in range(0, self.stackLen):
            self.f2 = of.sampleFrame(self.video)
            if self.f2 is None:
                raise StopIteration
            frame = of.normalizeOF(of.calcOF(self.f1, self.f2))
            self.xbuffer[i] = frame[..., 0]
            self.ybuffer[i] = frame[..., 1]
            self.f1 = self.f2
        return self.xbuffer, self.ybuffer


def cv2ToPIL(img):
    return Image.fromarray(img)

net = motion.MotionNet().cpu().eval()
net.load_state_dict(torch.load(os.path.join(motion.COMMON_PREFIX, "model.pkl")))

def most_frequent(List): 
    return max(set(List), key = List.count)

def predict(video):
    loader = OFloader(video)
    results = []
    for x, y in loader:
        stack = [motion.imgTrans(cv2ToPIL(z)) for z in x + y]
        tensor = torch.cat(stack, 0).unsqueeze(0)
        pred = net(tensor).view(6).cpu().detach()
        klass = torch.max(pred, 0)[1].item()
        results.append(klass)
    print("Predict vector is: {0}, final prediction is: {1}".format(results, most_frequent(results)))
