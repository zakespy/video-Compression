import cv2
import numpy as np
import pandas as pd
import subprocess
import os
from scipy.stats import entropy


def get_frame_types(video_filename):
    """
    Extract frame types (I, P, B) from a video file using FFprobe.
    :param video_filename: Name of the video file (assumed to be in the same folder as this script)
    :return: List of frame types (e.g., ['I', 'P', 'B', ...])
    """
    video_path = os.path.join(os.getcwd(), video_filename)

    # FFprobe command
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


# Initialize video capture
video = cv2.VideoCapture('sample.mp4')
# video = cv2.VideoCapture('samplevideo2.MOV')
success, frame = video.read()

if not success:
    print("Failed to read video file.")
    exit()

# Resize initial frame
frame = cv2.resize(frame, (640, 480))

# Initialize variables
pixelDiff = []
entropylist = []
pixelVar = []
edges = []
video_abs_diff=[]
frametype = get_frame_types('sample.mp4')  # Extract frame types once
# frametype = get_frame_types('samplevideo2.MOV')  # Extract frame types once
frame_index = 0

while success:
    # Read the next frame
    success, next_frame = video.read()
    if not success:
        break

    # Resize the next frame to match dimensions
    next_frame = cv2.resize(next_frame, (640, 480))
    
    # Pixel difference
    abs_diff = cv2.absdiff(frame, next_frame)
    video_abs_diff.append(abs_diff)
    mean_diff = np.mean(abs_diff)
    pixelDiff.append(mean_diff)

    # Shannon entropy
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    hist, _ = np.histogram(gray_frame.ravel(), bins=256, range=(0, 256))
    hist = hist / np.sum(hist)  # Normalize histogram
    entropy_val = entropy(hist, base=2)
    entropylist.append(entropy_val)
    
    # pixel variance intensity 
    pixelIntensities = gray_frame.flatten()
    pixelVar.append(np.var(pixelIntensities))
    
    # canny edge detection 
    edges_canny = cv2.Canny(gray_frame, 10, 20)
    # edge_sobel = cv2.Sobel(gray_frame, cv2.CV_8U, 1, 0, ksize=3)
    # edge_sobel = cv2.Sobel(next_frame, cv2.CV_8U, 1, 0, ksize=3)
    edges.append(edges_canny.flatten())
    
    cv2.imshow('Canny Edges', mean_diff)
    # cv2.imshow('gray', gray_frame)

    # # Print frame type and entropy
    # if frame_index < len(frametype):  # Ensure index is within bounds
    #     print(f"Frame {frame_index + 1}: Type = {frametype[frame_index]}, Entropy = {entropy_val:.4f}")
    
    # Update frame
    frame = next_frame
    frame_index += 1
    
    if cv2.waitKey(3) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# Print final results
# print("\nPixel Differences:", pixelDiff)
# print("\nEntropy List:", entropylist)
# print("\nPixel Intensity variance List:", pixelVar)
# print("\n Edges:", edges)

# pixelDiff = []
# entropylist = []
# pixelVar = []
# edges = []
# video_abs_diff=[]

# entropylist.pop(0)
# pixelVar.pop(0)   
# edges.pop(0)
# frametype.pop(0)
frameIndexList = [i for i in range(0,len(frametype)+1)]
data_tuples = list(zip(frameIndexList,pixelDiff,entropylist,pixelVar,[float(np.mean(i)) if isinstance(i, np.ndarray) else float(i) for i in edges],[float(np.mean(i)) if isinstance(i, np.ndarray) else float(i) for i in video_abs_diff],frametype))
df = pd.DataFrame(data_tuples,columns=['frame_index','pixel_diff',"entropy","pixel_intensity_var","edge","absolute_diff","frame_type"])
df.to_csv('video_frame_data.csv', encoding='utf-8', index=False, header=True)
print(df)