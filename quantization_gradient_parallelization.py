import torch
import torch.nn.functional as F
import numpy as np
import pickle
from scipy.fftpack import dct, idct
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from concurrent.futures import ProcessPoolExecutor
from motionVector_blockMatching import * 

QUANTI_BLOCK_SIZE = 8
QUANTIZATION_MATRIX = torch.tensor([[16,11,10,16,24,40,51,61],
                                    [12,12,14,19,26,58,60,55],
                                    [14,13,16,24,40,57,69,56],
                                    [14,17,22,29,51,87,80,62],
                                    [18,22,37,56,68,109,103,77],
                                    [24,35,55,64,81,104,113,92],
                                    [49,64,78,87,103,121,120,101],
                                    [72,92,95,98,112,100,103,99]], dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mse(a, b):
    return F.mse_loss(a, b)


def blockwise_dct(blocks):
    return torch.real(torch.fft.fft2(blocks))

def blockwise_idct(blocks):
    return torch.real(torch.fft.ifft2(blocks))


def extract_blocks(image):
    image = image.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    unfold = torch.nn.Unfold(kernel_size=QUANTI_BLOCK_SIZE, stride=QUANTI_BLOCK_SIZE)
    blocks = unfold(image).transpose(1, 2)  # [1, N, 64]
    blocks = blocks.view(1, -1, QUANTI_BLOCK_SIZE, QUANTI_BLOCK_SIZE)  # [1, N, 8, 8]
    return blocks


def reconstruct_image(blocks, height, width):
    blocks = blocks.view(1, -1, QUANTI_BLOCK_SIZE * QUANTI_BLOCK_SIZE).transpose(1, 2)
    fold = torch.nn.Fold(output_size=(height, width), kernel_size=QUANTI_BLOCK_SIZE, stride=QUANTI_BLOCK_SIZE)
    ones = torch.ones_like(blocks)
    image = fold(blocks) / fold(ones)
    return image.squeeze()


def gradient_optimization(frame_np, epochs):
    frame = torch.tensor(frame_np, dtype=torch.float32, device=device)
    height, width = frame.shape
    blocks = extract_blocks(frame)
    N = blocks.shape[1]

    alpha = torch.ones((N,), device=device, requires_grad=False)
    alpha_grad = torch.zeros_like(alpha)

    base_quant = QUANTIZATION_MATRIX.to(device).flatten()
    lr = 0.01
    delta = 3e-3

    for epoch in range(epochs):
        total_loss = 0
        alpha_grad.zero_()

        for idx in range(N):
            block = blocks[0, idx]
            dct_block = blockwise_dct(block)
            scale = alpha[idx]
            scaled_q = scale * base_quant.view(QUANTI_BLOCK_SIZE, QUANTI_BLOCK_SIZE)
            quantized = torch.round(dct_block / scaled_q)
            dequantized = quantized * scaled_q
            recon_block = blockwise_idct(dequantized)
            loss = mse(recon_block, block)

            alpha[idx] += delta
            scaled_q_d = alpha[idx] * base_quant.view(QUANTI_BLOCK_SIZE, QUANTI_BLOCK_SIZE)
            quantized_d = torch.round(dct_block / scaled_q_d)
            dequantized_d = quantized_d * scaled_q_d
            recon_block_d = blockwise_idct(dequantized_d)
            loss_d = mse(recon_block_d, block)

            grad = (loss_d - loss) / delta
            alpha[idx] -= delta

            alpha_grad[idx] = grad
            total_loss += loss.item()

        alpha -= lr * alpha_grad
        alpha.clamp_(0.1, 5.0)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return alpha.cpu().numpy().reshape((height // QUANTI_BLOCK_SIZE, width // QUANTI_BLOCK_SIZE))


def parallel_gradient_optimization(frames, epochs):
    with ProcessPoolExecutor() as executor:
        return list(executor.map(lambda frame: gradient_optimization(frame, epochs), frames))


def results(original, reconstructed):
    original = original.astype(np.float32)
    reconstructed = reconstructed.astype(np.float32)
    data_range = 255.0
    psnr = compare_psnr(original, reconstructed, data_range=data_range)
    ssim, _ = compare_ssim(original, reconstructed, data_range=data_range, full=True)
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")


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


with open('frames.pkl', 'rb') as file:
    # Load the data from the file
    frames = pickle.load(file)

frameType = getFrametype("SampleVideo.mp4")

Iframes = [frame for index, frame in enumerate(frames) if frameType[index] == "I"]

# using scaled quantization 
quant_scale = [gradient_optimization(frame,10) for frame in Iframes]
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

# Example usage:
# with open('frames.pkl', 'rb') as file:
#     frames = pickle.load(file)
# Iframes = [frame for i, frame in enumerate(frames) if frameType[i] == "I"]
# quant_scales = parallel_gradient_optimization(Iframes, 10)
# (Follow up with quantization/dequantization steps as needed.)
