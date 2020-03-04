import matplotlib.pyplot as plt
import pickle
import argparse

def plotInference(path):
    with open(path, 'rb') as handle:
        infer, truth = pickle.load(handle)

    fig = plt.figure()
    ax = fig.add_subplot(211)
    plt.grid(True)
    plt.title('Inference')
    xdata = list(range(len(infer)))
    bx = fig.add_subplot(212)
    ax.plot(xdata, infer, 'ro', label='infer', markersize=5)
    bx.plot(xdata, truth, 'bo', label='truth', markersize=5, alpha=0.5)
    #ax.legend(fancybox='True', shadow='True')
    plt.grid(True)
    plt.title('Truth')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="唐大傻的痴呆工具箱")
    parser.add_argument('--path', help='File path')
    args = parser.parse_args()

    plotInference(args.path)
