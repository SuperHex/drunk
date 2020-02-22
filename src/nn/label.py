
def buildLabelMap(path):
    f = open(path, 'r')
    lines = [[tag for tag in line.strip('\t\n\r').split(' ') if tag != ''] for line in f.readlines()]
    f.close()
    
    labels = dict()
    for video, start, end in lines:
        video_index = int(video)
        if video_index not in labels:
            labels[video_index] = set()
        for frame in range(int(start), int(end) + 1):
            labels[video_index].add(frame)
    #print(labels)
    return labels