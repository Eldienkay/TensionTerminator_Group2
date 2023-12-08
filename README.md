
# Technical Report: TensionTerminator_Group2

## Introduction

This technical report covers the TensionTerminator_Group2 project, specifically focusing on the `main.py` script, which is designed to analyze video files for specific human movements, count these movements, and display the count and other relevant information on the video.

## Installation and Configuration

### Prerequisites

- Python: version 3.8 - 3.11. (because of MediaPipe)
- Libraries: OpenCV (`cv2`), MediaPipe, NumPy, Tkinter
- A valid video file in a supported format (e.g., `.mp4`, `.avi`)

### Installation Steps

1. Clone the repository from GitHub:
   ```
   git clone https://github.com/Eldienkay/TensionTerminator_Group2.git
   git checkout Duo_balls_distance
   ```

2. Install required Python libraries:
   ```
   pip install opencv-python mediapipe numpy
   ```

3. Ensure a compatible video file is placed in the `videos` directory or choose a file at runtime.

## Functionality and Requirements

### Overview

`main.py` performs the following functions:

- Processes a selected video file to detect human movements (up and down movements).
- Counts these movements and displays the count in real-time on the video.
- Integrates with `movement_data.json` to store and potentially display movement data. Green font and real time remark for (almost) right up- and down movements, orange font and remark for almost right movement. Text can altered to concrete countings (e.g. up - and down movements >= 5 and <= 11 are right, too fast movements are wrong)

### Technical Specifications

- **Movement Detection**: Uses MediaPipe's pose estimation to detect human poses in each frame of the video.
- **Movement Counting**: A specific algorithm detects up and down movements and increments a counter accordingly.
- **Data Integration**: Movement data is written to `movement_data.json`, allowing for persistence and later analysis.

### System Architecture

- **main.py**: Main script handling video processing, pose estimation, movement detection, and display.
- **HelperFunctions.py**: Contains supplementary functions that assist `main.py`.
- **movement_data.json**: Stores data about detected movements.

### Interfaces and Modules

- **MediaPipe**: Used for pose estimation.
- **OpenCV**: Handles video file operations and adds text overlays to the video.
- **Tkinter**: Provides a simple GUI for selecting a video file.

### Application Parameters

- The reference pixel width, actual object width, and distance in `main.py` can be adjusted for different video conditions.
- The threshold values for movement detection in `HelperFunctions.py` can be modified to alter sensitivity.

## Usage Guidelines

### Running the Code

1. Run `main.py`:
   ```
   python main.py
   ```
2. Select a video file when prompted.
3. Observe the video playback with the movement count displayed.

### Integration into Third-Party Code

- Import the necessary functions from `main.py` or `HelperFunctions.py`.
- Adjust the parameters and function calls as per the requirement of the third-party project.
- Ensure that MediaPipe and OpenCV dependencies are met in the third-party project.

### Modifying the Code

- To alter movement detection logic, modify the relevant sections in `HelperFunctions.py` or the movement_data.json.
- movement_data.json is the exchange file for the app and can be adapted 
- For changes in how movement data is displayed or stored, update the respective sections in `main.py`.
