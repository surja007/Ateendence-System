import cv2
import numpy as np
import os
from datetime import datetime

def train_face():
    # Initialize face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Ask for the name
    name = input("Enter name for the new face: ")
    if not name:
        print("Name cannot be empty!")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera")
        return
    
    print(f"Starting training for {name}...")
    print("Please show your face from different angles and with different expressions.")
    print("The system will collect 20 samples.")
    
    samples = []
    sample_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract face region
            face_region = gray[y:y+h, x:x+w]
            
            # Resize face region to a fixed size
            face_region = cv2.resize(face_region, (100, 100))
            
            # Add sample if we haven't collected enough
            if sample_count < 20:
                samples.append(face_region)
                sample_count += 1
                print(f"Collected {sample_count}/20 samples")
                
                # Show the frame with sample count
                cv2.putText(frame, f"Samples: {sample_count}/20", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
        # Display the frame
        cv2.imshow('Training', frame)
        
        # Break if we've collected enough samples
        if sample_count >= 20:
            break
            
        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    if sample_count >= 20:
        try:
            # Save the samples
            samples_array = np.array(samples)
            np.save('training_samples.npy', samples_array)
            
            # Save the labels (all 0s since we're training one person)
            labels = np.zeros(len(samples), dtype=int)
            np.save('training_labels.npy', labels)
            
            # Save the name
            with open('known_names.txt', 'w') as f:
                f.write(f"{name}\n")
            
            print(f"Successfully collected {sample_count} samples for {name}")
            print("Training complete! You can now use the attendance system.")
            
        except Exception as e:
            print(f"Error saving training data: {str(e)}")
    else:
        print("Not enough samples collected!")
    
    # Initialize face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Ask for the name
    name = input("Enter name for the new face: ")
    if not name:
        print("Name cannot be empty!")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera")
        return
    
    print(f"Starting training for {name}...")
    print("Please show your face from different angles and with different expressions.")
    print("The system will collect 20 samples.")
    
    samples = []
    labels = []
    sample_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract face region
            face_region = gray[y:y+h, x:x+w]
            
            # Add sample if we haven't collected enough
            if sample_count < 20:
                samples.append(face_region)
                labels.append(0)  # Using 0 as label since we're training one person
                sample_count += 1
                print(f"Collected {sample_count}/20 samples")
                
                # Show the frame with sample count
                cv2.putText(frame, f"Samples: {sample_count}/20", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
        # Display the frame
        cv2.imshow('Training', frame)
        
        # Break if we've collected enough samples
        if sample_count >= 20:
            break
            
        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    if sample_count >= 20:
        try:
            # Train the recognizer
            face_recognizer.train(samples, np.array(labels))
            
            # Save the model
            face_recognizer.save('face_model.yml')
            
            # Save the name
            with open('known_names.txt', 'w') as f:
                f.write(f"{name}\n")
            
            print(f"Successfully trained {sample_count} samples for {name}")
            print("Training complete! You can now use the attendance system.")
            
        except Exception as e:
            print(f"Error saving training data: {str(e)}")
    else:
        print("Not enough samples collected!")

if __name__ == "__main__":
    train_face()
