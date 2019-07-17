import numpy as np
import cv2 as cv
import os
import time

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
    return flow

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

def sampleFrame(video):
    # skip one frame per sample to reduce fps to 12 (originally 25)
    frame = getFrame(video)
    skipNFrames(video, 1)
    return frame

def calcOF(frame1, frame2):
    return cv.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def calcSaveOFVideo(name, prefix):
    optical_flow_path = "OpticalFlows"
    of_abs_path = os.path.join(prefix, optical_flow_path)
    if not os.path.exists(of_abs_path):
        os.mkdir(of_abs_path)
    of_video_dir_path = os.path.join(of_abs_path, os.path.splitext(name)[0])
    if not os.path.exists(of_video_dir_path):
        os.mkdir(of_video_dir_path)
        video = cv.VideoCapture(os.path.join(os.path.join(prefix, "AVIClips"), name))
        prev = sampleFrame(video)
        sample = sampleFrame(video)
        count = 0
        start_time = time.time()
        while prev is not None and sample is not None:
            of = calcOF(prev, sample)
            prev = sample
            sample = sampleFrame(video)
            of_name_x = "OF" + str(count).zfill(4) + "_x.jpeg"
            of_name_y = "OF" + str(count).zfill(4) + "_y.jpeg"
            normalOF = normalizeOF(of)
            cv.imwrite(os.path.join(of_video_dir_path, of_name_x), normalOF[..., 0])
            cv.imwrite(os.path.join(of_video_dir_path, of_name_y), normalOF[..., 1])
            #print("wrote frame {0} into {1}".format(count, of_video_dir_path))
            count += 1
        end_time = time.time()
        print("Proceed {:d} frames in {:s}. Time used: {:.2f} s".format(count, of_video_dir_path, end_time - start_time))
        return count
    else:
        print("Skip existed clip {:s}".format(name))
        return None

def genOpticalFlowDir(folder, prefix):
    work_dir = os.path.join(prefix, folder)
    start = time.time()
    for file in os.listdir(work_dir):
        if file.endswith(".avi"):
            print("Processing video {:s}".format(file))
            calcSaveOFVideo(file, prefix)
    end = time.time()
    print("All done. Time used: {:.2f} s".format(end - start))

prefix = "/run/media/cirno/40127CD9E5B9A466/dataset/Hollywood2/"
avi_path = "AVIClips"

if __name__ == "__main__":
    genOpticalFlowDir(avi_path, prefix)
