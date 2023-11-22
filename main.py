from HelperFunctions import *


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
left_shoulder_buffer = deque(maxlen=20)
right_shoulder_buffer = deque(maxlen=20)


def process_frame(frame, calibration_data):
    # Convert the BGR image to RGB.
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the image and find pose landmarks.
    result = pose.process(image_rgb)

    # Draw the pose landmarks on the image.
    image_with_pose = frame.copy()

    if result.pose_landmarks:

        # Extract landmarks
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

        # Calculate the size of the head in pixels
        head_size_pixels = calculate_head_size_in_pixels(result.pose_landmarks.landmark, frame)

        # Estimate distance using the calibration data
        distance = estimate_distance(head_size_pixels, calibration_data)

        if distance >= 2:
            distance = 2

        # Display the distance information
        cv2.putText(frame, f"Distance: {distance:.2f} meters", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame


def calculate_head_size_in_pixels(landmarks, frame):
    # Example using NOSE and SHOULDER landmarks
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    # Estimate the top of the head as a point above the NOSE
    estimated_top_of_head_x = nose.x
    estimated_top_of_head_y = nose.y - (abs(left_shoulder.y - right_shoulder.y) * 0.5)

    # Convert these landmarks to pixel coordinates
    nose_x = int(nose.x * frame.shape[1])
    nose_y = int(nose.y * frame.shape[0])
    top_of_head_x = int(estimated_top_of_head_x * frame.shape[1])
    top_of_head_y = int(estimated_top_of_head_y * frame.shape[0])

    # Calculate the size (e.g., height) of the head in pixels
    head_size_pixels = np.sqrt((nose_x - top_of_head_x)**2 + (nose_y - top_of_head_y)**2)

    return head_size_pixels


def estimate_distance(ref_size_pixels, calibration_data):
    known_distance, known_size_pixels = calibration_data

    # To prevent division by zero
    if ref_size_pixels == 0:
        return float('inf')

    distance = (known_size_pixels / ref_size_pixels) * known_distance
    return distance


def main():
    cap = cv2.VideoCapture('videos/duo_balls_1.mp4')

    calibration_data = (1.0, 30)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame with MediaPipe to get pose landmarks
        frame_with_pose = process_frame(frame, calibration_data)

        # Display the frame with pose landmarks
        cv2.imshow('Frame', frame_with_pose)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
