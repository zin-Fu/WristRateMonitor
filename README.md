# Heart Rate Monitoring Application

This is a heart rate monitoring application with a graphical user interface (GUI) built using PyQt5 and OpenCV. It captures images from a webcam or video file and calculates the heart rate by processing the images.

## Features

- Capture real-time video from a webcam or load a video file.
- Process the captured frames to detect the user's pulse rate.
- Display the real-time heart rate on the GUI.
- Allow the user to adjust settings such as video source, region of interest, and processing parameters.

## Requirements

- numpy
- opencv-python
- PyQt5
- pyqtgraph
- scipy

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/zin-Fu/WristRateMonitor.git
   ```

2. Install the required dependencies:

   ```shell
   pip install -r requirements.txt
   ```

## Usage

1. Run the application by executing the `GUI.py` script:

   ```shell
   python GUI.py
   ```

2. The GUI window will open, providing options to select the video source (webcam or video file) and adjust the settings.

3. Click the "Start" button to begin capturing and processing the video frames.

4. The heart rate will be calculated and displayed in real-time on the GUI.

## Code Structure

The project consists of the following files:

- `GUI.py`: Contains the graphical user interface (GUI) implementation using PyQt5 and OpenCV.
- `interface.py`: Handles the image processing and data visualization using the OpenCV library.
- `process.py`: Detects the pulse and calculates the heart rate. It utilizes OpenCV for image processing, NumPy for data handling, and SciPy for signal processing.
- `signal_process.py`: Contains a signal processing class with various methods for data analysis and processing.
- `video.py`: Handles the processing of video files.
- `webcam.py`: Handles the processing of video streams from a webcam.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

