from compression_blockMatching import *
import numpy as np
import subprocess
import pickle


BLOCK_SIZE = 16
# NO_OF_WORKERS = os.cpu_count()
NO_OF_WORKERS = 2
SCALE_FACTOR = 2
frameType = getFrametype("sample.mp4")
# frames, scaled_up_motionVector = vector_motion_extractor('sample.mp4')


with open('motion_vector.pkl', 'rb') as file:
    # Load the data from the file
    scaled_up_motionVector = pickle.load(file)
    
with open('frames.pkl', 'rb') as file:
    # Load the data from the file
    frames = pickle.load(file)


residuals = []
predicted_frames = []


def prediction(prev_frame, motionVector, frameType):
    
    if(frameType == "I"):
        return None
    newFrame = prev_frame.copy()
    
    height,width = prev_frame.shape
    # print(prev_frame.shape)

    for i in range(0, height,BLOCK_SIZE):
        for j in range(0, width,BLOCK_SIZE):
            # print(motionVector[i])
            mov = motionVector[i*BLOCK_SIZE + j+1]
            # print(mov)
            # ymov = BLOCK_SIZE*motionVector[i][j][1]
            newFrame[i+mov[0]:i+BLOCK_SIZE +mov[0] , j+mov[1]:j+BLOCK_SIZE+mov[1]] = prev_frame[i:i+BLOCK_SIZE,j:j+BLOCK_SIZE] 
    
    return newFrame
    

# print(len(frames))
# print(frames[0].shape)
# print(len(scaled_up_motionVector[0]))
# print(len(scaled_up_motionVector))

for i in range(1, len(frames)):
    extra  = len(scaled_up_motionVector[i]) - (frames[i].shape[0]*frames[i].shape[1])//(BLOCK_SIZE**2)
    scaled_up_motionVector[i] = scaled_up_motionVector[i][:-(extra)]
    print((frames[i].shape[0]*frames[i].shape[1])//(BLOCK_SIZE**2))
    print(len(scaled_up_motionVector[i]))
    fr = prediction(frames[i-1], scaled_up_motionVector[i],frameType[i])
    if(fr is None):
        residuals.append(frames[i])
        predicted_frames.append(frames[i])
    else:
        residuals.append(fr-frames[i])
        predicted_frames.append(fr)
        
displayFrames(predicted_frames)
