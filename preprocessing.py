import cv2
import numpy as np



motionVectorWindow = 5

video = cv2.VideoCapture('SampleVideo_1280x720_1mb.mp4')
success, frame = video.read()

if not success:
    print("Error: Could not read the video file.")
    exit()

# Resize the initial frame and convert to grayscale
frame = cv2.resize(frame, (640, 480))
prev_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

motion_vector = []

def getMotionVector(prev_gray_scale_frame, next_gray_scale_frame):
    global motionVectorWindow
    # Ensure both frames are the same size and single-channel
    assert prev_gray_scale_frame.shape == next_gray_scale_frame.shape, "Frame sizes do not match."
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray_scale_frame, next_gray_scale_frame, None,
        pyr_scale=0.5, levels=3, winsize=motionVectorWindow,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow

while True:
    ret, next_frame = video.read()  # Read the next frame
    if not ret:
        break

    # Resize the next frame to match dimensions
    next_frame = cv2.resize(next_frame, (640, 480))

    # Convert the next frame to grayscale
    next_gray_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Calculate the motion vector
    motion = getMotionVector(prev_gray_frame, next_gray_frame)
    motion_vector.append(motion)


    # visualization 1 ********************************************************************
    # # Visualize the motion vectors
    # hsv = np.zeros((next_gray_frame.shape[0], next_gray_frame.shape[1], 3), dtype=np.uint8)
    # hsv[..., 1] = 255  # Set saturation to maximum

    # # Convert flow to polar coordinates (magnitude and angle)
    # magnitude, angle = cv2.cartToPolar(motion[..., 0], motion[..., 1])
    # hsv[..., 0] = angle * 180 / np.pi / 2  # Set hue based on flow angle
    # hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # # Convert HSV image to BGR for display
    # flow_visualization = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # cv2.imshow('Optical Flow', flow_visualization)


    #visuallization 2**********************************************************************
    # step = motionVectorWindow  # Grid spacing for arrows
    # for y in range(0, motion.shape[0], step):
    #     for x in range(0, motion.shape[1], step):
    #         dx, dy = motion[y, x]  # Get motion vector at this point
    #         start_point = (x, y)
    #         end_point = (int(x + dx), int(y + dy))
    #         color = (0, 255, 0)  # Green arrows
    #         thickness = 1
    #         cv2.arrowedLine(next_frame, start_point, end_point, color, thickness, tipLength=0.5)

    # # Display the frame with motion arrows
    # cv2.imshow('Motion Vectors', next_frame)


    # # Update the previous frame
    prev_gray_frame = next_gray_frame
    
    # # print(motion)

    # # Break the loop if 'q' is pressed
    # if cv2.waitKey(3) & 0xFF == ord('q'):
    #     break

# Release the video capture object and close the display window
video.release()
cv2.destroyAllWindows()
