import cv2
import mediapipe as mp
from collections import deque
import numpy as np


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
pose = mp_pose.Pose()

# Create deques to store the last 10 positions for each shoulder
left_shoulder_buffer = deque(maxlen=8)
right_shoulder_buffer = deque(maxlen=8)


def process_frame(frame):
    # Convert the BGR image to RGB.
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the image and find pose landmarks.
    result = pose.process(image_rgb)

    # Draw the pose landmarks on the image.
    image_with_pose = frame.copy()

    if result.pose_landmarks:
        left_shoulder = result.pose_landmarks.landmark[11]
        right_shoulder = result.pose_landmarks.landmark[12]

        # Convert the landmark positions to pixel coordinates
        ls_x = int(left_shoulder.x * frame.shape[1])
        ls_y = int(left_shoulder.y * frame.shape[0])
        rs_x = int(right_shoulder.x * frame.shape[1])
        rs_y = int(right_shoulder.y * frame.shape[0])

        # Add the current positions to the buffers
        left_shoulder_buffer.append((ls_x, ls_y))
        right_shoulder_buffer.append((rs_x, rs_y))

        # Calculate the average positions over the last 10 frames
        avg_left_shoulder = np.mean(left_shoulder_buffer, axis=0).astype(int)
        avg_right_shoulder = np.mean(right_shoulder_buffer, axis=0).astype(int)

        # Draw circles on the left and right shoulder landmarks
        cv2.circle(image_with_pose, (avg_left_shoulder[0], avg_left_shoulder[1]), 5, (0, 255, 0), -1)
        cv2.circle(image_with_pose, (avg_right_shoulder[0], avg_right_shoulder[1]), 5, (0, 255, 0), -1)

        # Draw a line between the left and right shoulder landmarks
        cv2.line(image_with_pose, (avg_left_shoulder[0], avg_left_shoulder[1]),
                 (avg_right_shoulder[0], avg_right_shoulder[1]), (0, 0, 255), 2)

    return image_with_pose


def main():
    cap = cv2.VideoCapture('videos/RGB_video1.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame with MediaPipe to get pose landmarks
        frame_with_pose = process_frame(frame)

        # Display the frame with pose landmarks
        cv2.imshow('Frame', frame_with_pose)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
