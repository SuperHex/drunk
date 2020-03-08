import wave
import numpy as np
import sys
import os
import argparse
import random

def loadInference(path):
    with open(path, 'r') as handle:
        inferStr = handle.readline()
        truthStr = handle.readline()
        infer = [int(i) for i in inferStr if i != '\n']
        truth = [int(i) for i in truthStr]
        return infer, truth

def deltaZeros(delta):
    return b'\0' * delta

def loadAllAudio(path):
    _, _, files = next(os.walk(path))
    outputs = []
    params = []
    for f in files:
        fullpath = os.path.join(path, f)
        with wave.open(fullpath, 'rb') as audio:
            params.append(audio.getparams()[:4])
            outputs.append(audio.readframes(audio.getnframes()))
    return outputs, params

def genShuffleList(size):
    choices = list(range(0, size))
    random.shuffle(choices)
    return choices

def generateWAVwith(infer, path, outPath, videoFrameRate):

    #sample = wave.open(path, 'rb')
    #channels, bytewidth, framerate, nframes = sample.getparams()[:4]
    #print(channels, bytewidth, framerate, nframes)
    #sample_data = sample.readframes(nframes)
    #sample.close()
    outputs, params = loadAllAudio(path)

    numAudios = len(outputs)
    choices = genShuffleList(numAudios)

    # write 0 to first frame
    zero = deltaZeros(48000 // videoFrameRate * 2 * 3)
    data = zero

    index = 0
    length = len(infer)
    while index < length:
        nextFrame = index + 1
        if nextFrame < length and infer[index] == 1 and infer[nextFrame] == 0:
            if len(choices) < 1:
                choices = genShuffleList(numAudios)
            choice = choices.pop()
            print("choose audio " + str(choice))
            channels, bytewidth, framerate, nframes = params[choice]
            sampleLen = nframes // (framerate // videoFrameRate)
            data += outputs[choice]
            index += sampleLen
        else:
            data += zero
            index += 1

    audio = wave.open(outPath, 'wb')
    audio.setnchannels(2)
    audio.setsampwidth(3)
    audio.setframerate(48000)
    audio.writeframes(data)
    audio.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='粪工具')
    parser.add_argument('-a', dest='audio', help='Audio folder path')
    parser.add_argument('-c', dest='infer', help='Inference file path')
    parser.add_argument('-o', dest='out_path', help='Output file path')

    args = parser.parse_args()

    infer, truth = loadInference(args.infer)
    generateWAVwith(infer, args.audio, args.out_path, 24)