import json
import os
import tkinter as tk
from tkinter import filedialog
import cv2
import mediapipe as mp
from collections import deque
import numpy as np

# Path to the JSON file
json_file_path = 'movement_data.json'


class MovingAverageFilter:
    def __init__(self, window_size):
        self.window_size = window_size
        self.data = []

    def update(self, value):
        self.data.append(value)
        if len(self.data) > self.window_size:
            self.data.pop(0)
        return sum(self.data) / len(self.data)


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)


def select_video_file():
    """Open a dialog to select a video file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window
    file_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video Files", "*.mp4 *.avi")],
        initialdir='./videos'
    )
    return file_path


def calculate_focal_length(pixel_width_ref, actual_width, distance_ref):
    # Calculate the focal length based on the reference image.
    focal_length = (pixel_width_ref * distance_ref) / actual_width
    return focal_length


def estimate_distance(pixel_width_current, actual_width, focal_length):
    # Estimate the distance of the object from the camera in the current frame.
    if pixel_width_current == 0:
        return float('inf')  # Avoid division by zero

    distance = (actual_width * focal_length) / pixel_width_current
    return distance


# Given data
pixel_width_ref = 500  # pixel width in reference image
average_head_width = 30
actual_width_ref = 30 # actual width of the object (in centimeters)
distance_ref = 70  # distance from camera in reference image (in centimeters)

# Calculate the focal length using the reference image
focal_length = calculate_focal_length(pixel_width_ref, actual_width_ref, distance_ref)

# Example usage
# Suppose in a current frame, the pixel width of the object is 500 pixels
pixel_width_current = 500

# Estimate the distance
estimated_distance = estimate_distance(pixel_width_current, actual_width_ref, focal_length)


def detect_head_top(frame):
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose detection
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Landmarks for shoulders
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Landmarks for eyes to approximate forehead position
        left_eye = landmarks[
            mp_pose.PoseLandmark.LEFT_EYE_INNER] if 'LEFT_EYE_INNER' in mp_pose.PoseLandmark.__members__ else landmarks[
            mp_pose.PoseLandmark.LEFT_EYE]
        right_eye = landmarks[
            mp_pose.PoseLandmark.RIGHT_EYE_INNER] if 'RIGHT_EYE_INNER' in mp_pose.PoseLandmark.__members__ else \
        landmarks[mp_pose.PoseLandmark.RIGHT_EYE]

        # Calculate midpoint between eyes as an approximation for the forehead
        forehead_x = (left_eye.x + right_eye.x) / 2
        forehead_y = (left_eye.y + right_eye.y) / 2

        # Calculate centroid using these landmarks
        head_x = int((left_shoulder.x + right_shoulder.x + forehead_x) / 3 * frame.shape[1])
        head_y = int((left_shoulder.y + right_shoulder.y + forehead_y) / 3 * frame.shape[0])

        # Adjusted head width (can be refined)
        head_width = int(abs(left_shoulder.x - right_shoulder.x) * frame.shape[1])

        # Draw the center of the head
        cv2.circle(frame, (head_x, head_y), 5, (0, 255, 0), -1)

        return head_x, head_y, head_width

    return None, None, None


def draw_landmarks_with_annotations(frame, results):
    if results.pose_landmarks:
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            # Calculate the x and y coordinates
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            # Draw a small circle at the landmark position
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Annotate the landmark
            landmark_name = mp_pose.PoseLandmark(id).name
            cv2.putText(frame, landmark_name, (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


def main():
    # Initialize a list to store the last 3 measurements
    recent_measurements = deque(maxlen=5)
    frame_counter = 0
    last_known_distance = None
    movement_count = 0
    frames_since_last_movement = 0
    current_state = 'up'  # Initial state
    movement_frame_counts = []
    movement_data = {"elements": []}  # JSON structure
    latest_movement_result = {"text": "", "color": "white"}
    consecutive_frames_over_100 = 0
    consecutive_frames_under_70 = 0

    video_path = select_video_file()

    if not video_path:
        print("No video file selected.")
        return

    # Now we use video_path in VideoCapture
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect the top of the head and get its pixel width
        head_x, head_y, head_width_pixels = detect_head_top(frame)

        # Check if head_width_pixels is within the specified range
        if head_width_pixels is not None:
            # Check if head_width_pixels is within the specified range to avoid temporary inaccuracies to falsify the data
            if head_width_pixels < 250:
                head_width_pixels = 250
            elif head_width_pixels > 500:
                head_width_pixels = 500
            # Add the measurement to the deque
            recent_measurements.append(head_width_pixels)

            # Update the distance every 3 frames
            if frame_counter % 1 == 0 and len(recent_measurements) == recent_measurements.maxlen:
                delayed_width_pixels = recent_measurements[0]
                last_known_distance = estimate_distance(delayed_width_pixels, actual_width_ref, focal_length)

        # Inside your main loop or function where you update the state
        if last_known_distance is not None:
            if last_known_distance > 100:
                consecutive_frames_over_100 += 1
                consecutive_frames_under_70 = 0  # Reset this counter as the condition is not met
            elif last_known_distance < 90:
                consecutive_frames_under_70 += 1
                consecutive_frames_over_100 = 0  # Reset this counter as the condition is not met
            else:
                # Reset both counters if neither condition is met
                consecutive_frames_over_100 = 0
                consecutive_frames_under_70 = 0

            # Now update the state transitions
            if current_state == 'up' and consecutive_frames_over_100 >= 2 and frames_since_last_movement >= 14:
                current_state = 'down'
                # Do not reset consecutive_frames_over_100 here
            elif current_state == 'down' and consecutive_frames_under_70 >= 2 and frames_since_last_movement >= 28:
                current_state = 'up'
                movement_count += 1  # Increment movement count on a full up movement
                movement_frame_counts.append(frames_since_last_movement)
                # Reset consecutive_frames_under_70 here if you want to require 6 new frames under 70 after transitioning to 'up'
                consecutive_frames_under_70 = 0

                # Evaluate the movement speed and update JSON object
                # Between 36 and 72 frames is considered good speed to us (between 1.2 and 2.4 seconds with 30 frames7s videos)
                if frames_since_last_movement < 37:
                    latest_result = {"text": "Bewege dich langsamer!", "rating": "0", "color": "red"}
                elif frames_since_last_movement > 71:
                    latest_result = {"text": "Fast perfekt!", "rating": "0.5", "color": "orange"}
                else:
                    latest_result = {"text": "Gut gemacht, weiter so!", "rating": "1", "color": "green"}

                movement_data["elements"].append(latest_result)
                latest_movement_result = latest_result
                frames_since_last_movement = 0

        # Display the movement count and frame count since last movement
        cv2.putText(frame, f"Up/Down Movements: {movement_count}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Display the latest movement result in the video
        if latest_movement_result["text"]:
            color = (0, 0, 0)  # Default to black
            if latest_movement_result["color"] == "red":
                color = (0, 0, 255)
            elif latest_movement_result["color"] == "green":
                color = (0, 255, 0)
            elif latest_movement_result["color"] == "orange":
                color = (0, 165, 255)

            cv2.putText(frame, latest_movement_result["text"], (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Frame', frame)

        frame_counter += 1
        frames_since_last_movement += 1

        # Show the frame
        cv2.imshow('Frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Read and display the content of the JSON file
    with open('movement_data.json', 'w') as json_file:
        json.dump(movement_data, json_file, indent=4)


if __name__ == "__main__":
    main()
