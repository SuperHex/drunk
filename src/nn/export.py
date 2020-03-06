import torch
import torchvision
import sys
from models import MotionNet

if __name__ == '__main__':
    name = sys.argv[1]
    output = sys.argv[2]
    net = MotionNet(6)
    net.load_state_dict(torch.load(name))
    net.eval()
    example = torch.rand(1, 6, 224, 224)
    tracedModule = torch.jit.trace(net, example)
    tracedModule.save(output)