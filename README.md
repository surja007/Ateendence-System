# Face Recognition Attendance System

A Python-based attendance system that uses face recognition to mark attendance. The system captures faces, trains a model to recognize them, and provides a graphical user interface for managing attendance.

## Features

- **Face Registration**: Register new faces with names
- **Face Recognition**: Real-time face detection and recognition
- **Attendance Tracking**: Automatically records attendance with timestamps
- **GUI Interface**: User-friendly interface for managing the system
- **CSV Export**: Saves attendance records in CSV format

## Prerequisites

- Python 3.7 or higher
- Webcam
- Required Python packages (see [Installation](#installation))

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd Attendence
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Train the System with New Faces

1. Create a folder named `images` in the project directory if it doesn't exist.
2. Inside the `images` folder, create a subfolder for each person with their name.
3. Place clear face images of each person in their respective folders.
4. Run the training script:
   ```bash
   python train_face.py
   ```
   Or use the GUI's "Train New Face" button.

### 2. Run the Attendance System

```bash
python attendance_gui.py
```

### 3. Using the GUI

- **Start/Stop Camera**: Toggle the camera feed
- **Mark Attendance**: Click to manually mark attendance for recognized faces
- **View Attendance**: View the attendance records in the table
- **Export CSV**: Save the attendance records to a CSV file
- **Train New Face**: Add new faces to the system

## Project Structure

- `attendance_gui.py`: Main application with GUI interface
- `face.py`: Core face recognition logic
- `train_face.py`: Script for training new faces
- `train_faces.py`: Alternative training script
- `deploy.prototxt`: Model configuration file
- `res10_300x300_ssd_iter_140000.caffemodel`: Pre-trained face detection model
- `images/`: Directory containing training images
- `attendance.csv`: Stores attendance records
- `known_faces.npy`: Encoded face data
- `known_names.txt`: List of known names
- `training_samples.npy` & `training_labels.npy`: Training data

## Dependencies

- OpenCV (`opencv-python`)
- face-recognition
- NumPy
- Pillow
- dlib

## Notes

- For best results, ensure good lighting when capturing face images
- The system works best with frontal face images
- Keep the face clearly visible and avoid obstructions

## Troubleshooting

- If you encounter issues with dlib installation, try:
  ```bash
  pip install cmake
  pip install dlib
  ```
- Ensure your webcam is properly connected and accessible
- Make sure you have sufficient disk space for storing face encodings and attendance records

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Uses OpenCV for computer vision tasks
- Built with Python's Tkinter for the GUI
- Face recognition powered by dlib and face-recognition library
