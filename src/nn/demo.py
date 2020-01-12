import optical_flow as of
import motion
import cv2 as cv
import torch
from PIL import Image
import os
import sys


def cv2ToPIL(img):
    return Image.fromarray(img, mode="L")

class OFloader:
    def __init__(self, path):
        self.stackLen = 20
        self.video = cv.VideoCapture(path)
        self.xbuffer = [None] * self.stackLen
        self.ybuffer = [None] * self.stackLen

    def __iter__(self):
        self.f1 = of.sampleFrame(self.video, skip=0)
        if self.f1 is None:
            raise StopIteration
        return self

    def __next__(self):
        for i in range(0, self.stackLen):
            f2 = of.sampleFrame(self.video, skip=0)
            if f2 is None:
                raise StopIteration
            frame = of.normalizeOF(of.calcOF(self.f1, f2))
            self.xbuffer[i] = frame[..., 0]
            self.ybuffer[i] = frame[..., 1]
            self.f1 = f2
            # path = os.path.join(motion.COMMON_PREFIX, "oftest")
            # print(self.xbuffer[i].dtype)
            # cv.imshow("image", self.xbuffer[i])
            # cv2ToPIL(self.xbuffer[i]).show()
            # cv.waitKey(0)
        return self.xbuffer, self.ybuffer

net = motion.MotionNet()
net.load_state_dict(torch.load(os.path.join(motion.COMMON_PREFIX, "model.pkl")))
net.cpu()
net.eval()

def most_frequent(List): 
    return max(set(List), key = List.count)

def predict(shape, video):
    loader = OFloader(video)
    results = []
    for x, y in loader:
        stack = [motion.imgTrans(cv2ToPIL(z)) for z in x + y]
        tensor = torch.cat(stack, 0).unsqueeze(0)
        pred = net(tensor).view([shape]).cpu().detach()
        print(pred)
        klass = torch.max(pred, 0)[1].item()
        results.append(klass)
    print("Predict vector is: {0}, final prediction is: {1}".format(results, most_frequent(results)))

if __name__ == "__main__":
    predict(int(sys.argv[1]), sys.argv[2])
