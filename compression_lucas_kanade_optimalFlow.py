import pandas as pd
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import convolve
from scipy.linalg import lstsq
from multiprocessing import Pool, cpu_count
import os

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

def gradient(frame1, frame2):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])

    gradient_x = convolve(frame2, sobel_x, mode='reflect')
    gradient_y = convolve(frame2, sobel_y, mode='reflect')
    gradient_t = frame2 - frame1
    return gradient_x, gradient_y, gradient_t

def compute_flow_for_region(args):
    """Compute optical flow for a single region."""
    window_I_x, window_I_y, window_I_t = args
    A = np.stack((window_I_x.flatten(), window_I_y.flatten()), axis=-1)
    b = -window_I_t.flatten()

    if np.linalg.cond(A.T @ A) < 1 / np.finfo(float).eps:  # Check for singularity
        v, _, _, _ = lstsq(A, b)  # Solve least squares
        return v[0], v[1]
    return 0, 0

def compute_optical_flow(I_x, I_y, I_t, window_size):
    half_window = window_size // 2
    h, w = I_x.shape

    # Prepare padded gradients
    padded_I_x = np.pad(I_x, half_window, mode='constant', constant_values=0)
    padded_I_y = np.pad(I_y, half_window, mode='constant', constant_values=0)
    padded_I_t = np.pad(I_t, half_window, mode='constant', constant_values=0)

    flow_u = np.zeros_like(I_x)
    flow_v = np.zeros_like(I_y)

    # Create tasks for parallel computation
    tasks = []
    for y in range(half_window, h + half_window):
        for x in range(half_window, w + half_window):
            window_I_x = padded_I_x[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
            window_I_y = padded_I_y[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
            window_I_t = padded_I_t[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
            tasks.append((window_I_x, window_I_y, window_I_t))

    # Parallel computation
    with Pool(cpu_count()) as pool:
        results = pool.map(compute_flow_for_region, tasks)

    # Map results back to flow matrices
    idx = 0
    for y in range(half_window, h + half_window):
        for x in range(half_window, w + half_window):
            flow_u[y - half_window, x - half_window], flow_v[y - half_window, x - half_window] = results[idx]
            idx += 1

    return flow_u, flow_v

def get_motion_vectors(frames, window_size=5):
    flow_u_list, flow_v_list = [], []

    for i in range(1, len(frames)):
        print(f"Processing frame pair {i - 1} -> {i}")
        I_x, I_y, I_t = gradient(frames[i - 1], frames[i])
        print("Done gradient")
        flow_u, flow_v = compute_optical_flow(I_x, I_y, I_t, window_size)
        print("Done flow")
        flow_u_list.append(flow_u)
        flow_v_list.append(flow_v)

    return flow_u_list, flow_v_list

def displayFlow(Flow_U_list, Flow_V_list, frames):
    fig, ax = plt.subplots()
    fig.figsize = (10, 10)
    ax.axis('off')

    def update(frame):
        img.set_data(frame)
        return img

    for i in range(0, len(frames)):
        if i == 0:
            img = ax.imshow(frames[i], cmap='gray')
        else:
            print(i)
            for y in range(0, Flow_U_list[i].shape[0], 5):
                for x in range(0, Flow_V_list[i].shape[1], 5):
                    u = Flow_U_list[i][y, x]
                    v = Flow_V_list[i][y, x]
                    plt.arrow(x, y, u, v, color='red', head_width=0.5, head_length=1)
        plt.gca().invert_yaxis()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ani = FuncAnimation(fig, update, frames, interval=3)
    plt.show()

frameType = getFrametype('sample.mp4')
frames = getFrames('sample.mp4')
print(type(frames[0]))
flow_u_list, flow_v_list = get_motion_vectors(frames, window_size=5)

print(flow_u_list[1].shape[0])
print(flow_u_list[1].shape[1])

# displayFlow(flow_u_list, flow_v_list, frames)
