import os

import cv2
import numpy as np

# Load the newly uploaded video in H.264 format
video_path = 'C:/Users/Daniel/OneDrive/Dokumente/GitHub/TensionTerminator_Group2/videos/Duo_balls_Lower_0.h264'
cap = cv2.VideoCapture(video_path)

print(video_path)

# Check if video loaded successfully
if not cap.isOpened():
    raise Exception("Error: Could not open video.")

# Initialize variables for movement detection
up_count = 0
down_count = 0
previous_frame = None
frame_diff_threshold = 50  # Threshold for detecting significant movement

# Function to calculate frame difference
def calculate_frame_difference(current_frame, previous_frame):
    diff = cv2.absdiff(current_frame, previous_frame)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return np.sum(thresh) / 255  # Count of changed pixels

# Analyze each frame in the video
while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no more frames

    # Convert to grayscale for easier analysis
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # If we have a previous frame to compare with
    if previous_frame is not None:
        frame_diff = calculate_frame_difference(gray_frame, previous_frame)

        # Detect significant movement
        if frame_diff > frame_diff_threshold:
            # Simple approach: alternate counting up and down
            if up_count == down_count:
                up_count += 1
            else:
                down_count += 1

    # Update the previous frame
    previous_frame = gray_frame

cap.release()

print(up_count, down_count)


