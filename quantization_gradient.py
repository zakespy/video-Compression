import pandas as pd 
import numpy as np
import pickle
import skimage.feature
from scipy.fft import dct
from scipy.fft import idct
from residuals import *
from motionVector_blockMatching import *
import torch
import torch.nn as nn
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
# import 


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

def mse(a,b):
    return np.mean((a-b)**2)

def gradientOptimization(frame,epochs):
    
    height,width = frame.shape
    alpha = np.ones((int(height/QUANTI_BLOCK_SIZE), int(width/QUANTI_BLOCK_SIZE)), dtype=np.float32)
    alpha_grad = np.zeros(alpha.shape, dtype=np.float32)
    
    lr = 0.01
    
    
    
    for epoch in range(epochs):
        
        totalLoss = 0
        alpha_grad.fill(0)
        
        for i in range(0, height, QUANTI_BLOCK_SIZE):
            for j in range(0, width, QUANTI_BLOCK_SIZE):
                bi, bj = i // QUANTI_BLOCK_SIZE, j // QUANTI_BLOCK_SIZE
                block = frame[i:i+QUANTI_BLOCK_SIZE, j:j+QUANTI_BLOCK_SIZE]

                DCT_block = dct(block, type=2)
                scaled_q = alpha[bi, bj] * QUANTIZATION_MATRIX
                quantized_block = np.round(DCT_block / scaled_q)

                dequantized_block = quantized_block * scaled_q
                reconBlock = idct(dequantized_block)

                loss = mse(reconBlock, block)
                totalLoss += loss

                delta = 3e-3
                alpha[bi, bj] += delta

                DCT_block_d = dct(block, type=2)
                scaled_q_d = alpha[bi, bj] * QUANTIZATION_MATRIX
                quantized_block_d = np.round(DCT_block_d / scaled_q_d)
                dequantized_block_d = quantized_block_d * scaled_q_d
                reconBlock_d = idct(dequantized_block_d)

                loss_d = mse(block, reconBlock_d)
                grad = (loss_d - loss) / delta

                alpha[bi, bj] -= delta
                alpha_grad[bi, bj] = grad

        
        alpha -= lr * alpha_grad
        alpha = np.clip(alpha,0.1,5.0)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {totalLoss:.4f}")
        
    return alpha 

            

def quantization(Iframe, quant_scale):
    height, width = Iframe.shape
    newIFrame = np.zeros(Iframe.shape)

    for i in range(0, height, QUANTI_BLOCK_SIZE):
        for j in range(0, width, QUANTI_BLOCK_SIZE):
            bi, bj = i // QUANTI_BLOCK_SIZE, j // QUANTI_BLOCK_SIZE
            block = Iframe[i:i+QUANTI_BLOCK_SIZE, j:j+QUANTI_BLOCK_SIZE]
            DCT_block = dct(block, type=2)

            scaled_q = quant_scale[bi, bj] * QUANTIZATION_MATRIX[:block.shape[0], :block.shape[1]]
            quantized_block = np.round(DCT_block / scaled_q)

            newIFrame[i:i+QUANTI_BLOCK_SIZE, j:j+QUANTI_BLOCK_SIZE] = quantized_block
    return newIFrame

def dequantization(Iframe, quant_scale):
    height, width = Iframe.shape
    newIFrame = np.zeros(Iframe.shape)

    for i in range(0, height, QUANTI_BLOCK_SIZE):
        for j in range(0, width, QUANTI_BLOCK_SIZE):
            bi, bj = i // QUANTI_BLOCK_SIZE, j // QUANTI_BLOCK_SIZE
            block = Iframe[i:i+QUANTI_BLOCK_SIZE, j:j+QUANTI_BLOCK_SIZE]

            scaled_q = quant_scale[bi, bj] * QUANTIZATION_MATRIX[:block.shape[0], :block.shape[1]]
            dequantizedBlock = block * scaled_q
            ogBlock = idct(dequantizedBlock, type=2)

            newIFrame[i:i+QUANTI_BLOCK_SIZE, j:j+QUANTI_BLOCK_SIZE] = ogBlock
    return newIFrame


def results(original, reconstructed):
    original = original.astype(np.float32)
    reconstructed = reconstructed.astype(np.float32)

    # PSNR and SSIM expect range info for float types
    data_range = 255.0  # Use 255 for 8-bit images

    psnr = compare_psnr(original, reconstructed, data_range=data_range)
    ssim, _ = compare_ssim(original, reconstructed, data_range=data_range, full=True)

    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")


Iframes = [frame for index, frame in enumerate(frames) if frameType[index] == "I"]

# using scaled quantization 
quant_scale = [gradientOptimization(frame,10) for frame in Iframes]
compressedIframes = [quantization(frame,quant_scale[index]) for index,frame in enumerate(Iframes)]
decompressedIframes = [dequantization(frame,quant_scale[index]) for index,frame in enumerate(compressedIframes)]

# using base quantization 
quant = [np.ones(quanta.shape,dtype=np.float32) for quanta in quant_scale]
norm_compressedIframes = [quantization(frame,quant[index]) for index,frame in enumerate(Iframes)]
norm_decompressedIframes = [dequantization(frame,quant[index]) for index,frame in enumerate(norm_compressedIframes)]

print("Results of scaled quantization ")
for og,rc in zip(Iframes,decompressedIframes):
    results(og,rc)

print("Result of Normal quantization ")
for og,rc in zip(Iframes,norm_decompressedIframes):
    results(og,rc)

print("starting to display images")

displayFrames(Iframes)

displayFrames(compressedIframes)
displayFrames(decompressedIframes)



displayFrames(norm_compressedIframes)
displayFrames(norm_decompressedIframes)

print("original first")
print(Iframes[0])

print("after compression")
print(compressedIframes[0])

print("after decompression ")
print(decompressedIframes[0])
