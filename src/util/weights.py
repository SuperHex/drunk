import sys

def parseLines(path):
    f = open(path, 'r')
    lines = [[tag for tag in line.strip('\t\n\r').split(' ') if tag != ''] for line in f.readlines()]
    f.close()
    return lines

def computeWeights(path):
    lines = parseLines(path)
    keyFrames, total = 0, 0
    for video, start, end in lines:
        keyFrames += 1
        for frame in range(int(start), int(end) + 1):
            total += 1
    return total, keyFrames

if __name__ == '__main__':
    t, f = computeWeights(sys.argv[1])
    print('Action: {0}, Key: {1}'.format(t, f))