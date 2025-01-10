from compression_blockMatching import *
import numpy as np
import subprocess
import pickle


BLOCK_SIZE = 16
# NO_OF_WORKERS = os.cpu_count()
NO_OF_WORKERS = 2
SCALE_FACTOR = 2
# frameType = getFrametype("sample.mp4")
# frames, scaled_up_motionVector = vector_motion_extractor('sample.mp4')


with open('motion_vector.pkl', 'rb') as file:
    # Load the data from the file
    scaled_up_motionVector = pickle.load(file)
    
with open('frames.pkl', 'rb') as file:
    # Load the data from the file
    frames = pickle.load(file)





def prediction(prev_frame, motionVector, frameType):
    
    if(frameType == "I"):
        return None
    newFrame = np.zeros(prev_frame.shape)
    # newFrame = prev_frame.copy()
    
    height,width = prev_frame.shape
    # print(prev_frame.shape)
    index = 0
    for i in range(0, height,BLOCK_SIZE):
        for j in range(0, width,BLOCK_SIZE):
            # print(motionVector[i])
            # MVWidth = width//BLOCK_SIZE
            # mov = motionVector[i*MVWidth + j]
            
            # mvi = (index // (width // BLOCK_SIZE)) * BLOCK_SIZE + BLOCK_SIZE // 2
            # mvj = (index % (width // BLOCK_SIZE)) * BLOCK_SIZE + BLOCK_SIZE // 2
            
            MVWidth = prev_frame.shape[1] // BLOCK_SIZE
            index = (i // BLOCK_SIZE) * MVWidth + (j // BLOCK_SIZE)
            mvi,mvj = motionVector[index]
            # print(mov)
            # ymov = BLOCK_SIZE*motionVector[i][j][1]
            
            # newFrame[i+mov[0]:i+BLOCK_SIZE +mov[0] , j+mov[1]:j+BLOCK_SIZE+mov[1]] = prev_frame[i:i+BLOCK_SIZE,j:j+BLOCK_SIZE]

            block = prev_frame[i:i+BLOCK_SIZE,j:j+BLOCK_SIZE]
            start_x = i+mvi
            start_y = j+mvj
            end_x = i+BLOCK_SIZE +mvi
            end_y = j+BLOCK_SIZE+mvj
            
            if (start_x < 0 or start_y < 0  or end_x > height or end_y > width):
                continue
            
            newFrame[start_x:end_x , start_y:end_y] = block
            
        index += 1    
    
    
    return newFrame
    

def predict_frame(ref_frame, motion_vectors,frameType):
    """
    Predict a frame based on motion vectors and a reference I-frame.

    Args:
        ref_frame (ndarray): The reference frame (I-frame) as a 2D NumPy array.
        motion_vectors (list): A list of tuples, where each tuple (dx, dy) is the motion vector for a block.
        block_size (int): The size of each block (e.g., 16 for 16x16 blocks).

    Returns:
        ndarray: The predicted frame as a 2D NumPy array.
    """
    
    if(frameType == "I"):
        return ref_frame
    
    block_size = BLOCK_SIZE
    height, width = ref_frame.shape
    predicted_frame = np.zeros_like(ref_frame)

    vector_idx = 0
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            # Ensure the motion vector index is valid
            if vector_idx >= len(motion_vectors):
                raise ValueError("Motion vector list is smaller than expected for the given frame size.")

            # Get the motion vector for the current block
            dx, dy = motion_vectors[vector_idx]

            # Compute the top-left corner of the corresponding block in the reference frame
            ref_x = i + dx
            ref_y = j + dy

            if ref_x < 0 or ref_y < 0 or ref_x + block_size > height or ref_y + block_size > width:
                continue
                # raise ValueError("Invalid motion vector ({}, {}) for block at ({}, {})".format(dx, dy, i, j))

            # Extract the block from the reference frame and handle boundary cases
            ref_block = ref_frame[
                max(0, ref_x):min(height, ref_x + block_size),
                max(0, ref_y):min(width, ref_y + block_size)
            ]

            # Place the block in the predicted frame (same location as the current block)
            predicted_frame[
                i:min(i + block_size, height),
                j:min(j + block_size, width)
            ] = ref_block

            # Increment the motion vector index
            vector_idx += 1

    return predicted_frame


def residuals_extractor(frames, scaled_up_motionVector, frameType):

    residuals = []
    predicted_frames = []

    residuals.append(np.zeros(frames[0].shape))
    predicted_frames.append(frames[0])

    for i in range(1, len(frames)):

        # fr = predict_frame(predicted_frames[i-1], scaled_up_motionVector[i-1],frameType[i])
        fr = prediction(predicted_frames[i-1], scaled_up_motionVector[i-1],frameType[i])

        if(fr is None):
            residuals.append(frames[i])
            predicted_frames.append(frames[i])
        else:
            residuals.append(frames[i] - fr)
            # residuals.append(fr-frames[i-1])
            predicted_frames.append(fr)


    diffFrames = []    
    for i in range(0,len(frames)):
        diffFrames.append(predicted_frames[i] + residuals[i])
    
    return residuals, predicted_frames, diffFrames
    
def interFramePredictor():
    pass 

if __name__ == "__main__":
    # frames, scaled_up_motionVector = vector_motion_extractor('sample.mp4')
    frameType = getFrametype("sample.mp4")
    residuals, predicted_frames, diffFrames = residuals_extractor(frames, scaled_up_motionVector, frameType)
    displayFrames(frames)
    displayFrames(predicted_frames)
    # displayFrames(residuals)
    displayFrames(diffFrames)

    __all__ = [ 'residuals_extractor' ,'prediction','predict_frame']