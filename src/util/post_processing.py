import matplotlib.pyplot as plt
import pickle
import argparse

def loadInference(path):
    with open(path, 'rb') as handle:
        infer, truth = pickle.load(handle)
        return infer, truth

def plotInference(infer, truth):
    xdata = list(range(len(infer)))
    fig = plt.figure()
    ax = fig.add_subplot(211)
    plt.grid(True)
    plt.title('Inference')
    plt.xlim(0, len(xdata))
    bx = fig.add_subplot(212)
    ax.plot(xdata, infer, 'ro', label='infer', markersize=5)
    bx.plot(xdata, truth, 'bo', label='truth', markersize=5, alpha=0.5)
    #ax.legend(fancybox='True', shadow='True')
    plt.grid(True)
    plt.title('Truth')
    plt.xlim(0, len(xdata))
    plt.show()

def flip(num):
    return 0 if num == 1 else 1

def removeSingleJitter(infer):
    previous = infer[0]
    for i in range(len(infer)):
        point = infer[i]
        if i == 0:
            continue
        if point != previous:
            if i + 1 < len(infer) and point != infer[i + 1]:
                infer[i] = flip(point)
        previous = point
    return infer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="唐大傻的痴呆工具箱")
    parser.add_argument('--path', help='File path')
    args = parser.parse_args()

    infer, truth = loadInference(args.path)
    plotInference(removeSingleJitter(infer), truth)
