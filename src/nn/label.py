
def parseLines(path):
    f = open(path, 'r')
    lines = [[tag for tag in line.strip('\t\n\r').split(' ') if tag != ''] for line in f.readlines()]
    f.close()
    return lines

def buildLabelMap(path):
    lines = parseLines(path)
    labels = dict()
    for video, start, end in lines:
        video_index = int(video)
        if video_index not in labels:
            labels[video_index] = set()
        for frame in range(int(start), int(end) + 1):
            labels[video_index].add(frame)
    #print(labels)
    return labels

def buildLabelBCE(path):
    lines = parseLines(path)
    labels = dict()
    labels['actioness'] = dict()
    labels['start'] = dict()
    labels['end'] = dict()
    for video, start, end in lines:
        video_index = int(video)
        if video_index not in labels['actioness']:
            labels['actioness'][video_index] = set()
            labels['start'][video_index] = set()
            labels['end'][video_index] = set()
        for frame in range(int(start), int(end) + 1):
            labels['actioness'][video_index].add(frame)
            labels['start'][video_index].add(frame)
            labels['end'][video_index].add(frame)
    #print(labels)
    return labels