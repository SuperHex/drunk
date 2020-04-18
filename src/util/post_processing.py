import matplotlib.pyplot as plt
import pickle
import argparse

def loadInference(path):
    with open(path, 'r') as handle:
        inferStr = handle.readline()
        truthStr = handle.readline()
        infer = [int(i) for i in inferStr]
        truth = [int(i) for i in truthStr]
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

def removeDuplicate(infer):
    TOO_CLOSE_WINDOW = 3
    EVENT_WINDOW = 3
    
    event_start, event_end, index = - EVENT_WINDOW, - TOO_CLOSE_WINDOW, 1
    while index < len(infer):
        current = infer[index]
        previous = infer[index - 1]
        # Rising edge
        if previous == 0 and current == 1:
            # check if two rising edge is too close
            print(index - event_end - 1)
            if abs(index - event_end - 1) < TOO_CLOSE_WINDOW:
                infer[index] = 0
            else:
                event_start = index
                print("start ", event_start)

        # Falling edge
        elif previous == 1 and current == 0:
            # also check if event is too short
            print(index - event_start)
            if abs(index - event_start) < EVENT_WINDOW:
                infer[index] = 1
            else:
                event_end = index - 1
                print("end ", event_end)


        index += 1

    return infer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="唐大傻的痴呆工具箱")
    parser.add_argument('--path', help='File path')
    args = parser.parse_args()

    infer, truth = loadInference(args.path)
    
    infer = removeSingleJitter(infer)
    infer = removeDuplicate(infer)
    
    plotInference(infer, truth)
