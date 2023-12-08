import cv2
import mediapipe as mp
from collections import deque
import numpy as np

from main import mp_pose


def ears_are_visible(landmarks):
    # Assuming landmarks for left and right ears are available in MediaPipe
    left_ear = landmarks[mp.solutions.pose.PoseLandmark.LEFT_EAR]
    right_ear = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EAR]

    # Check if the visibility score of the ears is above a certain threshold
    visibility_threshold = 0.5  # Adjust this threshold based on your needs
    return left_ear.visibility > visibility_threshold and right_ear.visibility > visibility_threshold


def calculate_distance(landmark1, landmark2, frame):
    x1, y1 = int(landmark1.x * frame.shape[1]), int(landmark1.y * frame.shape[0])
    x2, y2 = int(landmark2.x * frame.shape[1]), int(landmark2.y * frame.shape[0])
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def calculate_head_dimensions(landmarks, frame):
    head_width = None
    head_height = None

    # Check if ear landmarks are visible and use them for width
    if ears_are_visible(landmarks):
        left_ear = landmarks[mp.solutions.pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EAR]
        head_width = calculate_distance(left_ear, right_ear, frame)

    # TODO: Implement calculate_width_using_alternative_landmarks
    # If ears are not visible, use alternative landmarks for width
    if head_width is None:
        head_width = calculate_width_using_alternative_landmarks(landmarks, frame)

    # TODO: Implement calculate_head_height
    # Calculate height of the head
    head_height = calculate_head_height(landmarks, frame)

    return head_width, head_height


def calculate_width_using_alternative_landmarks(landmarks, frame):
    # Example: Using temples or sides of the forehead
    # Replace these landmarks with the ones that are most suitable for your setup
    left_forehead = landmarks[mp.solutions.pose.PoseLandmark.LEFT_EYE_OUTER]
    right_forehead = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EYE_OUTER]

    return calculate_distance(left_forehead, right_forehead, frame)


def calculate_head_height(landmarks, frame):
    # Example: Using forehead and upper neck
    # Replace these landmarks with the ones that are most suitable for your setup
    forehead = landmarks[mp.solutions.pose.PoseLandmark.FOREHEAD_GLABELLA]  # If available
    upper_neck = landmarks[mp.solutions.pose.PoseLandmark.NOSE_BRIDGE]  # An alternative landmark

    return calculate_distance(forehead, upper_neck, frame)




