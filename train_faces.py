import cv2
import os
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import sys
import pickle
from pathlib import Path

def train_faces():
    print("[INFO] Starting face recognition training...")
    
    # Path to images directory
    images_path = 'images'
    
    # Lists to store face templates and names
    known_faces: List[np.ndarray] = []
    known_names: List[str] = []
    
    try:
        # Initialize face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("[INFO] Face detector initialized successfully")
        
        # Process each person in the images directory
        for person_name in os.listdir(images_path):
            person_folder = os.path.join(images_path, person_name)
            print(f"[INFO] Processing person: {person_name}")
            
            # Process each image for the person
            for img_file in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_file)
                print(f"[INFO] Processing image: {img_path}")
                
                # Load the image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[WARNING] Could not load image: {img_path}")
                    continue
                
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                print(f"[INFO] Image converted to grayscale")
                
                # Detect faces
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                print(f"[INFO] Found {len(faces)} faces in image")
                
                if len(faces) > 0:
                    # Get the first face
                    (x, y, w, h) = faces[0]
                    face_region = img[y:y+h, x:x+w]
                    
                    # Resize face region to standard size
                    face_region = cv2.resize(face_region, (100, 100))
                    
                    known_faces.append(face_region)
                    known_names.append(person_name)
                    print(f"[INFO] Added face template for {person_name}")
                    print(f"[INFO] Face region dimensions: {face_region.shape}")
                else:
                    print(f"[WARNING] No face found in {img_path}")
        
        if len(known_faces) == 0:
            print("[ERROR] No faces were successfully processed")
            return
        
        # Save the trained model
        print("[INFO] Saving trained model...")
        np.save('known_faces.npy', known_faces)
        with open('known_names.txt', 'w') as f:
            for name in known_names:
                f.write(f"{name}\n")
        
        print("[INFO] Face recognition model trained successfully!")
        print(f"Total faces trained: {len(known_faces)}")
        print(f"Names trained: {known_names}")
        
    except Exception as e:
        print(f"[ERROR] Failed to train face recognition model: {str(e)}")
        import traceback
        print("[ERROR] Detailed error:")
        print(traceback.format_exc())

if __name__ == "__main__":
    train_faces()
