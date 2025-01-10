import pandas as pd 
import numpy as np
from residuals import *
import pickle
from motionVector_blockMatching import *



with open('frames.pkl', 'rb') as file:
    # Load the data from the file
    frames = pickle.load(file)
    
frameType = getFrametype("sample.mp4")


def interFrameCompressor(Iframes):
    
    pass
        


Iframes = [frame for index, frame in enumerate(frames) if frameType[index] == "I"]

compressedIframes = interFrameCompressor(Iframes)


#  Pending will be done later after implementation of video compression and decompression using quantization 
