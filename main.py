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
pixel_width_ref = 300  # pixel width in reference image
average_head_width = 16
actual_width_ref = 16  # actual width of the object (in centimeters)
distance_ref = 70  # distance from camera in reference image (in centimeters)




# Calculate the focal length using the reference image
focal_length = calculate_focal_length(pixel_width_ref, actual_width_ref, distance_ref)

# Example usage
# Suppose in a current frame, the pixel width of the object is 200 pixels
pixel_width_current = 0

# Estimate the distance
estimated_distance = estimate_distance(pixel_width_current, actual_width_ref, focal_length)


def detect_head_top(frame):
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose detection
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Original ear landmarks
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]

        # Adjust these landmarks slightly outward
        adjustment_factor = 0.1  # This factor determines how much the landmarks are adjusted outward
        adjusted_left_x = left_ear.x - adjustment_factor * abs(left_ear.x - right_ear.x)
        adjusted_right_x = right_ear.x + adjustment_factor * abs(left_ear.x - right_ear.x)

        # Calculate the center position for the top of the head
        head_x = int((adjusted_left_x + adjusted_right_x) / 2 * frame.shape[1])
        head_y = int((left_ear.y + right_ear.y) / 2 * frame.shape[0])

        # Calculate the adjusted width of the head
        head_width = int(abs(adjusted_left_x - adjusted_right_x) * frame.shape[1])

        # Draw for visualization
        '''cv2.circle(frame, (head_x, head_y), 5, (0, 255, 0), -1)
        cv2.line(frame, (int(adjusted_left_x * frame.shape[1]), int(left_ear.y * frame.shape[0])),
                 (int(adjusted_right_x * frame.shape[1]), int(right_ear.y * frame.shape[0])), (255, 0, 0), 2)'''

        return head_x, head_y, head_width

    return None, None, None


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
            if video_path == "C:/Users/Daniel/OneDrive/Dokumente/GitHub/TensionTerminator_Group2/videos/Duo_balls_Lower_0.mp4":
                if head_width_pixels < 170:
                    head_width_pixels = 170
                elif head_width_pixels > 340:
                    head_width_pixels = 340
            # Add the measurement to the deque
            recent_measurements.append(head_width_pixels)

            # Update the distance every 3 frames
            if frame_counter % 5 == 0 and len(recent_measurements) == recent_measurements.maxlen:
                delayed_width_pixels = recent_measurements[0]
                last_known_distance = estimate_distance(delayed_width_pixels, actual_width_ref, focal_length)

        # State machine for counting up and down movements
        if last_known_distance is not None:
            if video_path == "C:/Users/Daniel/OneDrive/Dokumente/GitHub/TensionTerminator_Group2/videos/Duo_balls_Lower_0.mp4":
                if current_state == 'up' and last_known_distance > 120:
                    current_state = 'down'
                    # Do not reset frames_since_last_movement here
                elif current_state == 'down' and last_known_distance < 90:
                    current_state = 'up'
                    movement_count += 1  # Increment movement count on a full up movement
                    movement_frame_counts.append(frames_since_last_movement)

                    # Evaluate the movement speed and update JSON object
                    if frames_since_last_movement < 35:
                        latest_result = {"text": "Bewege dich langsamer!", "rating": "0", "color": "red"}
                    elif frames_since_last_movement > 70:
                        latest_result = {"text": "Fast perfekt!", "rating": "0.5", "color": "orange"}
                    else:
                        latest_result = {"text": "Gut gemacht, weiter so!", "rating": "1", "color": "green"}

                    movement_data["elements"].append(latest_result)
                    latest_movement_result = latest_result
                    frames_since_last_movement = 0

            elif video_path == "C:/Users/Daniel/OneDrive/Dokumente/GitHub/TensionTerminator_Group2/videos/duo_balls_upper_01.mp4":
                if current_state == 'up' and last_known_distance > 400:
                    current_state = 'down'
                elif current_state == 'down' and last_known_distance < 250 and ((frames_since_last_movement > 48 and movement_count >= 1) or (frames_since_last_movement > 80 and movement_count == 0)):
                    current_state = 'up'
                    movement_count += 1  # Increment movement count on a full up movement
                    movement_frame_counts.append(frames_since_last_movement)

                    # Evaluate the movement speed and update JSON object
                    if frames_since_last_movement < 40:
                        latest_result = {"text": "Bewege dich langsamer!", "rating": "0", "color": "red"}
                    elif frames_since_last_movement > 70:
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
