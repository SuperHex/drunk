#! /usr/bin/python3

import numpy as np
import cv2 as cv
import os
from sys import argv
from time import time

def nextFrame(video):
    ret, frame = video.read()
    if ret:
        return frame
    else:
        None

def saveFrames(video_path, name, save_path, skip = 0):
    video = cv.VideoCapture(os.path.join(video_path, name))

    stem = name.split('.')[0]
    counter = 0
    folderPath = os.path.join(save_path, stem)
    if not os.path.exists(folderPath):
        os.mkdir(folderPath)

    start = time()
    while True:
        frame = nextFrame(video)
        if frame is not None:
            imageName = stem + "_RGB_" + str(counter).zfill(4) + ".jpeg"
            cv.imwrite(os.path.join(folderPath, imageName), frame)
            counter += 1
        else:
            break

        for _ in range(0, skip):
            nextFrame(video)
    end = time()
    print("Processed 1 video in {:.2f} s".format(end - start))


if __name__ == "__main__":
    if len(argv) < 5:
        print("Usage: saveFrames path [video name] [save to] [skip]")
    else:
        saveFrames(argv[1], argv[2], argv[3], int(argv[4]))