import numpy as np
import cv2 as cv
import os
import time
import argparse

def flow2BGR(flow):
    # for display flow image
    w, h, _ = flow.shape
    hsv = np.zeros((w, h, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2 / 360 * 255
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr

def normalizeOF(flow):
    bound = 15
    flow = np.round((flow + bound) / (2.0 * bound) * 255)
    flow[flow < 0] = 0
    flow[flow > 255] = 255
    return flow.astype(np.uint8)

def getFrame(video):
    ret, frame = video.read()
    if ret:
        return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    else:
        return None

def skipNFrames(video, n):
    status = True
    while status and (n > 0):
        status, _ = video.read()
        n -= 1

def sampleFrame(video, skip=1):
    # skip one frame per sample to reduce fps to 12 (originally 25)
    frame = getFrame(video)
    skipNFrames(video, skip)
    return frame

def calcOF(frame1, frame2):
    return cv.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def calcOpticalFlow(fullPath, outPath):
    video = cv.VideoCapture(fullPath)
    prev = sampleFrame(video, skip=0)
    sample = sampleFrame(video, skip=0)
    count = 0
    start_time = time.time()
    while prev is not None and sample is not None:
        of = calcOF(prev, sample)
        prev = sample
        sample = sampleFrame(video, skip=0)
        of_name_x = "OF" + str(count).zfill(4) + "_x.jpeg"
        of_name_y = "OF" + str(count).zfill(4) + "_y.jpeg"
        normalOF = normalizeOF(of)
        cv.imwrite(os.path.join(outPath, of_name_x), normalOF[..., 0])
        cv.imwrite(os.path.join(outPath, of_name_y), normalOF[..., 1])
        #print("wrote frame {0} into {1}".format(count, of_video_dir_path))
        count += 1
    end_time = time.time()
    print("Proceed {:d} frames in {:s}. Time used: {:.2f} s".format(count, outPath, end_time - start_time))
    return count

def calcSaveOFVideo(path, outPath, video):
    if not os.path.exists(outPath):
        os.mkdir(outPath)
    ofOutputDir = os.path.join(outPath, os.path.splitext(video)[0])
    if not os.path.exists(ofOutputDir):
        os.mkdir(ofOutputDir)
        calcOpticalFlow(os.path.join(path, video), ofOutputDir)
    else:
        print("Skip existed clip {:s}".format(ofOutputDir))
        return None

def genOpticalFlowDir(path, outPath, video=None):
    start = time.time()
    if video is None:
        for file in os.listdir(path):
            # if file.endswith(".avi") and not file.startswith("auto", 10, 14) and not file.startswith("test", 10, 14):
            if not file.startswith("auto", 10, 14) and not file.startswith("test", 10, 14):
                print("Processing video {:s}".format(file))
                calcSaveOFVideo(path, outPath, file)
    else:
        file = video
        if not file.startswith("auto", 10, 14) and not file.startswith("test", 10, 14):
            print("Processing video {:s}".format(file))
            calcSaveOFVideo(path, outPath, file)
    end = time.time()
    print("All done. Time used: {:.2f} s".format(end - start))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate optical flow")
    parser.add_argument('--path', help='Input video parent folder path')
    parser.add_argument('--video', default=None, help='Input video full name e.g. 01.avi')
    parser.add_argument('--output', help='Output parent folder')
    args = parser.parse_args()
    
    genOpticalFlowDir(args.path, args.output, args.video)
