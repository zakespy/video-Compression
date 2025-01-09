import pandas as pd
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import convolve
from scipy.linalg import lstsq
from multiprocessing import Pool, cpu_count
import pickle
import time
import os

BLOCK_SIZE = 16
# NO_OF_WORKERS = os.cpu_count()
NO_OF_WORKERS = 2
SCALE_FACTOR = 2


def getFrametype(videoFileName):
    video_path = os.path.join(os.getcwd(), videoFileName)
    cmd = [
        "ffprobe",
        "-show_frames",
        "-select_streams", "v",
        "-show_entries", "frame=pict_type",
        "-of", "csv",
        video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        frame_types = [line.split(",")[1].strip() for line in result.stdout.splitlines() if line.startswith("frame")]
        return frame_types
    except subprocess.CalledProcessError as e:
        print(f"Error running FFprobe: {e.stderr}")
        return []

def getFrames(videoFileName, frame_type='I'):
    video_path = os.path.join(os.getcwd(), videoFileName)
    
    # Get video resolution
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        video_path
    ]
    try:
        probe_result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        width, height = map(int, probe_result.stdout.strip().split(','))
    except subprocess.CalledProcessError as e:
        print(f"Error running FFprobe for resolution: {e.stderr}")
        return []

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", "select='not(eq(pict_type\\,{}))'".format(frame_type),
        "-vsync", "vfr",
        "-q:v", "2",
        "-f", "image2pipe",
        "-vcodec", "rawvideo",
        "-pix_fmt", "gray",
        "-"
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        frame_data = result.stdout
        frame_size = (height, width)  # Grayscale frames
        frame_data = np.frombuffer(frame_data, np.uint8)
        frame_data = frame_data.reshape(-1, height, width)
        return frame_data
    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg: {e.stderr}")
        return []

def displayFrames(frames):
    fig, ax = plt.subplots()
    ax.axis('off')
    im = ax.imshow(frames[0], cmap='gray', animated=True)
    def update(frame):
        im.set_array(frame)
        return [im]
    ani = FuncAnimation(fig, update, frames=frames, blit=True,interval=10)
    plt.show()
    
def displayFlow(frames, motionVector):
    fig, ax = plt.subplots()
    ax.axis('off')
    
    im = ax.imshow(frames[0], cmap='gray', animated=True)
    arrows = []  # To store arrow references for clearing

    def update(index):
        # Update the frame image
        im.set_array(frames[index])
        
        # Remove existing arrows
        for arrow in arrows:
            arrow.remove()
        arrows.clear()

        # Draw motion vectors for current frame (if not the first)
        if index != 0:
            height, width = frames[index].shape
            for idx, (mv_x, mv_y) in enumerate(motionVector[index-1]):
                i = (idx // (width // BLOCK_SIZE)) * BLOCK_SIZE + BLOCK_SIZE // 2
                j = (idx % (width // BLOCK_SIZE)) * BLOCK_SIZE + BLOCK_SIZE // 2
                arrow = ax.arrow(j, i, mv_y, mv_x, color='red', head_width=2, head_length=2)
                arrows.append(arrow)

        return [im] + arrows

    ani = FuncAnimation(fig, update, frames=len(frames), blit=False, interval=10)
    plt.show()
    
    
def display_upscaled_flow(frames,upscaled_motionVector):
    fig, ax = plt.subplots()
    ax.axis('off')
    
    im = ax.imshow(frames[0], cmap='gray', animated=True)
    arrows = []  # To store arrow references for clearing

    def update(index):
        # Update the frame image
        im.set_array(frames[index])
        
        # Remove existing arrows
        for arrow in arrows:
            arrow.remove()
        arrows.clear()

        # Draw motion vectors for current frame (if not the first)
        if index != 0:
            height, width = frames[index].shape
            upscaled_blocksize = BLOCK_SIZE*SCALE_FACTOR
            for idx, (mv_x, mv_y) in enumerate(upscaled_motionVector[index-1]):
                i = (idx // (width // upscaled_blocksize)) * upscaled_blocksize + upscaled_blocksize // 2
                j = (idx % (width // upscaled_blocksize)) * upscaled_blocksize + upscaled_blocksize // 2
                arrow = ax.arrow(j, i, mv_y, mv_x, color='red', head_width=2, head_length=2)
                arrows.append(arrow)

        return [im] + arrows

    ani = FuncAnimation(fig, update, frames=len(frames), blit=False, interval=10)
    plt.show()




# Parallel approach along with vectorizea approach 
def blockMatchingWorker(args):
    currFrame,refFrame,start_row, end_row = args
    blocksize = BLOCK_SIZE
    searchsize = 7
    
    height,width = currFrame.shape
    motionVector = []
    
    for i in range(start_row,end_row,blocksize):
        for j in range(0,width,blocksize):
            block = currFrame[i:i+blocksize,j:j+blocksize]
            
            bestMSE = float('inf')
            bestMatch = (0,0)
            
            # vectorized approach(better than exhaustive search)
            ref_x_range = range(max(0,i-searchsize),min(height-blocksize,i+searchsize+1))
            ref_y_range = range(max(0,j-searchsize),min(height-blocksize,j+searchsize+1))
            
            for ref_x in ref_x_range:
                for ref_y in ref_y_range:
                    ref_block = refFrame[ref_x:ref_x+blocksize,ref_y:ref_y+blocksize]
                    mse = np.mean((block-ref_block)**2)
                    if(mse<bestMSE):
                        bestMSE = mse
                        bestMatch = (ref_x-i,ref_y-j)
            
            motionVector.append(bestMatch)
    return motionVector
    

def blockmatching(currFrame,refFrame):
    height,_  = currFrame.shape
    num_workers = NO_OF_WORKERS
    pool = Pool(num_workers)
    
    
    rows_per_worker = height//num_workers
    args = [(currFrame,refFrame,i*rows_per_worker,(i+1)*rows_per_worker) for i in range(num_workers)]
    
    results = pool.map(blockMatchingWorker,args)
    pool.close()
    pool.join()
    
    motionVector = [mv for result in results for mv in result]
    return motionVector
        
def downscaleFrame(frame):
    new_height = frame.shape[0]//SCALE_FACTOR
    new_width = frame.shape[1]//SCALE_FACTOR
    
    return frame[:new_height*SCALE_FACTOR,:new_width*SCALE_FACTOR].reshape(new_height,SCALE_FACTOR,new_width,SCALE_FACTOR).mean(axis=(1,3))

def upscaleMV(motionVector):
    return [(int(dx*SCALE_FACTOR),int(dy*SCALE_FACTOR)) for dx,dy in motionVector]
# vectorized approach and old exhaustive search methods
# def blockmatching(currFrame,refFrame):
#     blocksize = BLOCK_SIZE
#     searchsize = 7
    
#     height,width = currFrame.shape
#     motionVector = []
    
#     for i in range(0,height,blocksize):
#         for j in range(0,width,blocksize):
#             block = currFrame[i:i+blocksize,j:j+blocksize]
            
#             bestMSE = float('inf')
#             bestMatch = (0,0)
            
#             # vectorized approach(better than exhaustive search)
#             ref_x_range = range(max(0,i-searchsize),min(height-blocksize,i+searchsize+1))
#             ref_y_range = range(max(0,j-searchsize),min(height-blocksize,j+searchsize+1))
            
#             for ref_x in ref_x_range:
#                 for ref_y in ref_y_range:
#                     ref_block = refFrame[ref_x:ref_x+blocksize,ref_y:ref_y+blocksize]
#                     mse = np.mean((block-ref_block)**2)
#                     if(mse<bestMSE):
#                         bestMSE = mse
#                         bestMatch = (ref_x-i,ref_y-j)
            
            
            
#             # old exhaustive search method
#             # for x in range(-searchsize,searchsize+1):
#             #     for y in range(-searchsize,searchsize+1):
#             #         ref_x = i+x
#             #         ref_y = j+y
                    
#             #         if(0<=ref_x<height-blocksize and 0<=ref_y<width-blocksize):
#             #             refBlock =  refFrame[ref_x:ref_x+blocksize,ref_y:ref_y+blocksize]
#             #             mse = np.mean((block-refBlock)**2)
                        
#             #             if(mse<bestMSE):
#             #                 bestMSE = mse
#             #                 bestMatch = (x,y)
            
#             motionVector.append(bestMatch)
#     return motionVector

def vector_motion_extractor(videoFileName):
    frameType = getFrametype(videoFileName)
    frames = getFrames(videoFileName)
    absFrame = [frames[i] - frames[i - 1] for i in range(1, len(frames))]

    stTime = time.time()
    scaled_down_frame = [downscaleFrame(frame) for frame in frames]

    print("blockingMatching started ")
    motionVector = [blockmatching(frames[i], frames[i - 1]) for i in range(1, len(frames))]
    # motionVector = [blockmatching(scaled_down_frame[i], scaled_down_frame[i - 1]) for i in range(1, len(frames))]
    print("blockingMatching Done ", time.time() - stTime)

    # scaled_up_motionVector = [upscaleMV(mv) for mv in motionVector]

    return frames,motionVector
    # return frames,scaled_up_motionVector

if __name__ == "__main__":
    # frames, scaled_up_motionVector =  vector_motion_extractor('sample.mp4')
    frames, motionVector =  vector_motion_extractor('sample.mp4')
    # displayFrames(frames)
    # displayFrames(absFrame)
    
    with open('motion_vector.pkl', 'wb') as file:
    # Serialize and write the data to the file
        pickle.dump(motionVector, file)
    with open('frames.pkl', 'wb') as file:
    # Serialize and write the data to the file
        pickle.dump(frames, file)
    
    displayFlow(frames, motionVector)
    # display_upscaled_flow(frames, scaled_up_motionVector)
    
    __all__ = [ 'vector_motion_extractor' ,'displayFlow','display_upscaled_flow','getFrametype','displayFrames','BLOCK_SIZE','SCALE_FACTOR','NO_OF_WORKERS']

            
    



# displayFlow(flow_u_list, flow_v_list, frames)
