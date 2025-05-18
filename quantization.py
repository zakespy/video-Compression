import pandas as pd 
import numpy as np
import pickle
import skimage.feature
from scipy.fft import dct
from scipy.fft import idct
from residuals import *
from motionVector_blockMatching import *


with open('frames.pkl', 'rb') as file:
    # Load the data from the file
    frames = pickle.load(file)
    
frameType = getFrametype("SampleVideo.mp4")
QUANTI_BLOCK_SIZE = 8
QUANTIZATION_MATRIX = np.array([[16,11,10,16,24,40,51,61],
                                [12,12,14,19,26,58,60,55],
                                [14,13,16,24,40,57,69,56],
                                [14,17,22,29,51,87,80,62],
                                [18,22,37,56,68,109,103,77],
                                [24,35,55,64,81,104,113,92],
                                [49,64,78,87,103,121,120,101],
                                [72,92,95,98,112,100,103,99]])
# QUANTIZATION_MATRIX = np.array([[17,  18,  24,  47,  99,  99,  99,  99,],[
#     18,  21,  26,  66,  99,  99,  99,  99,],[
#     24,  26,  56,  99,  99,  99,  99,  99,],[
#     47,  66,  99,  99,  99,  99,  99,  99,],[
#     99,  99,  99,  99,  99,  99,  99,  99,],[
#     99,  99,  99,  99,  99,  99,  99,  99,],[
#     99,  99,  99,  99,  99,  99,  99,  99,],[
#     99,  99,  99,  99,  99,  99,  99,  99]])


def scalingFactor( frame):
    height,width = frame.shape
    for i in range( 0,height,QUANTI_BLOCK_SIZE):
        for j in range ( 0,width,QUANTI_BLOCK_SIZE):
            block = frame[i:i+QUANTI_BLOCK_SIZE,j:j+QUANTI_BLOCK_SIZE]
            
            # DCT_block = dct(block,type=2)
            
            # variance
            variance = np.sum(np.square(block-np.mean(block)))/(BLOCK_SIZE**2)
            
            # edges Density
            edgesDen = len(skimage.feature.canny(image=block))
            
            # Entropy 
            marg =np.histogramdd(np.ravel(block),bins=256 )/(BLOCK_SIZE**2)
            marg  = list(filter(lambda p:p>0,np.ravel(marg)))
            entropy = -np.sum(np.multiply(marg, np.log2(marg)))
            
            # energy ratio
            
            
            
            

def quantization(Iframe):
    
    height,width = Iframe.shape
    newIFrame = np.zeros(Iframe.shape)
    for i in range( 0,height,QUANTI_BLOCK_SIZE):
        for j in range ( 0,width,QUANTI_BLOCK_SIZE):
            block = Iframe[i:i+QUANTI_BLOCK_SIZE,j:j+QUANTI_BLOCK_SIZE]
            DCT_block = dct(block,type=2)
            quantized_block = np.round(DCT_block/QUANTIZATION_MATRIX)
            newIFrame[i:i+QUANTI_BLOCK_SIZE,j:j+QUANTI_BLOCK_SIZE] = quantized_block
    return newIFrame

def dequantization(Iframe):
    height,width = Iframe.shape
    newIFrame = np.zeros(Iframe.shape)
    for i in range(0,height,QUANTI_BLOCK_SIZE):
        for j in range(0,width,QUANTI_BLOCK_SIZE):
            block = Iframe[i:min(i+QUANTI_BLOCK_SIZE, height), j:min(j+QUANTI_BLOCK_SIZE, width)]
            # dequantizedBlock = np.floor(block*QUANTIZATION_MATRIX)
            # dequantizedBlock = np.ceil(block*QUANTIZATION_MATRIX)
            dequantizedBlock = block * QUANTIZATION_MATRIX[:block.shape[0], :block.shape[1]]
            ogBlock = idct(dequantizedBlock, type=2)
            # newIFrame[i:i+QUANTI_BLOCK_SIZE,j:j+QUANTI_BLOCK_SIZE] = dequantizedBlock
            newIFrame[i:min(i+QUANTI_BLOCK_SIZE, height), j:min(j+QUANTI_BLOCK_SIZE, width)] = ogBlock
    return newIFrame


Iframes = [frame for index, frame in enumerate(frames) if frameType[index] == "I"]

compressedIframes = [quantization(frame) for frame in Iframes]
decompressedIframes = [dequantization(frame) for frame in compressedIframes]

displayFrames(Iframes)
displayFrames(compressedIframes)
displayFrames(decompressedIframes)

print("original first")
print(Iframes[0])

print("after compression")
print(compressedIframes[0])

print("after decompression ")
print(decompressedIframes[0])
