import tkinter as tk
from tkinter import ttk, messagebox
import threading
import cv2
import numpy as np
import os
from datetime import datetime
import time
from typing import Optional, Tuple
from PIL import Image, ImageTk

class AttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("800x600")
        
        # Initialize variables
        self.last_attendance_message = None
        self.attendance_message_timer = None
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Initialize face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize variables
        self.known_names = []
        self.training_mode = False
        self.training_name = None
        self.training_samples = []
        self.training_labels = []
        
        # Load known names
        if os.path.exists('known_names.txt'):
            with open('known_names.txt', 'r') as f:
                self.known_names = f.read().splitlines()
        
        # Initialize training variables
        self.training_mode = False
        self.training_label = None
        self.training_samples = []
        self.training_labels = []
        self.training_name = None
        
        # Status label
        self.status_label = ttk.Label(self.main_frame, text="Status: Ready", font=("Helvetica", 12))
        # Recognized person label
        self.recognized_label = ttk.Label(self.main_frame, text="Recognized: None", font=("Helvetica", 14, "bold"), foreground="blue")
        self.recognized_label.grid(row=0, column=2, pady=10)
        
        # Status label
        self.status_label = ttk.Label(self.main_frame, text="Status: Ready", font=("Helvetica", 12))
        self.status_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Age label
        self.age_label = ttk.Label(self.main_frame, text="Age: Not detected", font=("Helvetica", 12))
        self.age_label.grid(row=0, column=3, pady=10)
        
        # Start/Stop buttons
        self.start_button = ttk.Button(self.main_frame, text="Start Attendance", command=self.start_attendance)
        self.start_button.grid(row=1, column=0, pady=5)
        
        self.stop_button = ttk.Button(self.main_frame, text="Stop Attendance", command=self.stop_attendance, state=tk.DISABLED)
        self.stop_button.grid(row=1, column=1, pady=5)
        
        # Training button
        self.train_button = ttk.Button(self.main_frame, text="Train New Face", command=self.start_training)
        self.train_button.grid(row=1, column=2, pady=5)
        
        # Training status
        self.training_status = ttk.Label(self.main_frame, text="", font=("Helvetica", 10))
        self.training_status.grid(row=1, column=3, pady=5)
        
        # Camera feed display
        self.camera_frame = ttk.LabelFrame(self.main_frame, text="Camera Feed", padding="5")
        self.camera_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Face display
        self.face_label = ttk.Label(self.camera_frame)
        self.face_label.grid(row=0, column=0, pady=5)
        
        # Camera feed label
        self.camera_label = ttk.Label(self.camera_frame)
        self.camera_label.grid(row=1, column=0, pady=5)
        
        # Attendance log
        self.log_frame = ttk.LabelFrame(self.main_frame, text="Attendance Log", padding="5")
        self.log_frame.grid(row=3, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create text widget for log
        self.log_text = tk.Text(self.log_frame, height=10, width=80)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.log_frame, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text['yscrollcommand'] = scrollbar.set
        
        # Create attendance message label
        self.attendance_message = ttk.Label(self.main_frame, text="", font=("Helvetica", 14, "bold"), foreground="green")
        self.attendance_message.grid(row=4, column=0, columnspan=4, pady=10)
        
        # Camera status
        self.camera_running = False
        self.camera_thread = None
        self.last_recognized = None
        self.capture = None
        self.photo = None
        
        # Load known faces
        self.load_known_faces()
        
    def load_known_faces(self):
        try:
            # Load the face templates and names from the saved files
            self.known_faces = np.load('known_faces.npy')
            with open('known_names.txt', 'r') as f:
                self.known_names = [line.strip() for line in f.readlines()]
            
            print(f"[INFO] Loaded {len(self.known_faces)} face templates")
            print(f"[INFO] Known names: {self.known_names}")
            self.log_message(f"[INFO] Loaded {len(self.known_faces)} face templates")
            self.log_message(f"[INFO] Known names: {self.known_names}")
        except Exception as e:
            self.log_message(f"[ERROR] Failed to load known faces: {str(e)}")
            messagebox.showerror("Error", f"Failed to load known faces: {str(e)}")
    
    def start_attendance(self):
        if not self.camera_running:
            self.camera_running = True
            self.camera_thread = threading.Thread(target=self.run_camera)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
            self.start_button['state'] = tk.DISABLED
            self.stop_button['state'] = tk.NORMAL
            self.status_label['text'] = "Status: Running"
            self.training_status['text'] = ""
            self.training_mode = False
            
    def stop_attendance(self):
        if hasattr(self, 'camera_running') and self.camera_running:
            self.camera_running = False
            if hasattr(self, 'camera_thread') and self.camera_thread.is_alive():
                self.camera_thread.join()
            self.camera_label.configure(image='')
            self.recognized_label['text'] = ""
            self.age_label['text'] = ""
            self.training_status['text'] = ""
            self.training_mode = False
            self.start_button['state'] = tk.NORMAL
            self.stop_button['state'] = tk.DISABLED
            self.status_label['text'] = "Status: Stopped"
            self.training_status['text'] = ""
            self.training_mode = False
            
    def estimate_age(self, frame, face_location) -> Optional[int]:
        try:
            # Get face dimensions
            top, right, bottom, left = face_location
            face_width = right - left
            face_height = bottom - top

            # Estimate age based on face size and landmarks
            if face_width < 70 or face_height < 70:
                return 0  # Baby
            elif face_width < 90 or face_height < 90:
                return 5  # Toddler
            elif face_width < 110 or face_height < 110:
                return 10  # Child
            elif face_width < 130 or face_height < 130:
                return 15  # Teen
            elif face_width < 150 or face_height < 150:
                return 25  # Young adult
            elif face_width < 170 or face_height < 170:
                return 35  # Adult
            else:
                return 50  # Mature adult

        except Exception as e:
            print(f"Error estimating age: {str(e)}")
            return None

    def start_training(self):
        if not self.camera_running:
            self.training_name = tk.simpledialog.askstring("Input", "Enter name for the new face:")
            if self.training_name:
                self.start_attendance()
                self.training_mode = True
                self.training_status['text'] = f"Training mode: Collecting samples for {self.training_name}"
                self.training_samples = []
                self.training_labels = []

    def save_training_data(self):
        try:
            # Save the trained model
            self.face_recognizer.save('face_model.yml')
            
            # Save the names mapping
            with open('known_names.txt', 'a') as f:
                f.write(f"{self.training_name}\n")
            
            # Save the training samples
            np.save('training_samples.npy', np.array(self.training_samples))
            np.save('training_labels.npy', np.array(self.training_labels))
            
            self.log_message(f"[INFO] Successfully trained {len(self.training_samples)} samples for {self.training_name}")
            messagebox.showinfo("Success", f"Successfully trained {len(self.training_samples)} samples for {self.training_name}")
            
            # Reset training mode
            self.training_mode = False
            self.training_status['text'] = ""
            self.training_name = None
            self.training_samples = []
            self.training_labels = []
            
        except Exception as e:
            self.log_message(f"[ERROR] Failed to save training data: {str(e)}")
            messagebox.showerror("Error", f"Failed to save training data: {str(e)}")

    def run_camera(self):
        try:
            # Initialize camera
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                self.log_message("[ERROR] Could not open camera")
                self.stop_attendance()
                return

            # Set camera properties for better stability
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            
            self.camera_running = True
            
            # Initialize frame processing variables
            last_frame_time = time.time()
            frame_interval = 1/30  # Target 30 FPS
            last_frame = None
            
            while self.camera_running:
                try:
                    # Control frame rate
                    current_time = time.time()
                    if current_time - last_frame_time < frame_interval:
                        time.sleep(frame_interval - (current_time - last_frame_time))
                    last_frame_time = current_time

                    # Read frame
                    ret, frame = self.capture.read()
                    if not ret:
                        self.log_message("[ERROR] Could not read frame from camera")
                        self.stop_attendance()
                        break

                    # Convert to grayscale once
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Detect faces
                    faces = self.face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30)
                    )

                    # Clear labels if no face is detected
                    if len(faces) == 0:
                        self.recognized_label['text'] = "No face detected"
                        self.age_label['text'] = ""
                        self.last_recognized = ""
                        continue

                    # Process each detected face
                    for (x, y, w, h) in faces:
                        # Draw rectangle around face
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                        # Extract face region
                        face_region = gray[y:y+h, x:x+w]

                        # Resize face region to a fixed size
                        face_region = cv2.resize(face_region, (100, 100))

                        # Handle face recognition based on mode
                        if self.training_mode:
                            try:
                                # Add sample to training data
                                self.training_samples.append(face_region)
                                self.training_labels.append(len(self.known_names))  # Use current length as label
                                
                                # Update status
                                self.training_status['text'] = f"Training mode: Collected {len(self.training_samples)} samples for {self.training_name}"
                                
                                if len(self.training_samples) >= 20:  # Collect 20 samples
                                    self.save_training_data()
                                    self.stop_attendance()
                            except Exception as e:
                                print(f"[ERROR] Training error: {str(e)}")
                                self.training_status['text'] = f"Training error: {str(e)}"
                        else:
                            # For now, just show the face region
                            try:
                                # Convert to PIL image for display
                                face_image = Image.fromarray(face_region)
                                face_photo = ImageTk.PhotoImage(face_image)
                                
                                # Update the recognized label with face image
                                if hasattr(self, 'face_label'):
                                    self.face_label.configure(image=face_photo)
                                    self.face_label.image = face_photo
                                    self.recognized_label['text'] = "Face detected"
                                else:
                                    self.recognized_label['text'] = "Face detected"
                                    
                                # Try to estimate age
                                age = self.estimate_age(frame, (y, x, y+h, x+w))
                                if age:
                                    self.age_label['text'] = f"Age: {age}"
                            except Exception as e:
                                print(f"[ERROR] Face processing error: {str(e)}")
                                self.recognized_label['text'] = "Face detected"
                                self.age_label['text'] = ""

                    # Convert frame to tkinter format and update display
                    try:
                        # Convert BGR to RGB once
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Resize frame to fit display
                        height, width = rgb_frame.shape[:2]
                        max_size = 640
                        if width > height:
                            new_width = max_size
                            new_height = int(height * (max_size / width))
                        else:
                            new_height = max_size
                            new_width = int(width * (max_size / height))
                        
                        rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))
                        
                        # Convert to PIL image
                        image = Image.fromarray(rgb_frame)
                        photo = ImageTk.PhotoImage(image)
                        
                        # Update camera label
                        self.camera_label.configure(image=photo)
                        self.camera_label.image = photo
                        
                    except Exception as e:
                        print(f"[ERROR] Frame display error: {str(e)}")

                except Exception as e:
                    print(f"[ERROR] Frame processing error: {str(e)}")
                    continue

            # Cleanup
            self.camera_running = False
            self.capture.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            self.log_message(f"[ERROR] Camera error: {str(e)}")
            self.stop_attendance()

    def mark_attendance(self, name):
        try:
            # Get current date and time
            date = datetime.now().strftime("%Y-%m-%d")
            time = datetime.now().strftime("%H:%M:%S")
            
            # Write attendance to file
            with open('attendance.csv', 'a') as f:
                f.write(f"{name},{date},{time}\n")
            
            self.log_message(f"[INFO] Marked attendance for {name} at {time}")
            
        except Exception as e:
            self.log_message(f"[ERROR] Failed to mark attendance: {str(e)}")

    def log_message(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, full_message)
        self.log_text.see(tk.END)
        
        # Show success message for attendance
        if "recognized" in message.lower():
            self.show_attendance_success(message)

    def show_attendance_success(self, message):
        """Show a temporary success message for attendance."""
        if self.attendance_message_timer:
            self.root.after_cancel(self.attendance_message_timer)
            
        # Extract name from message
        name = message.split()[0]
        self.attendance_message['text'] = f" Attendance marked successfully: {name}"
        self.attendance_message['foreground'] = "green"
        
        # Clear the message after 3 seconds
        self.attendance_message_timer = self.root.after(3000, self.clear_attendance_message)

    def clear_attendance_message(self):
        """Clear the attendance success message."""
        self.attendance_message['text'] = ""
        self.attendance_message['foreground'] = "black"

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystem(root)
    root.mainloop()
