import face_recognition # type: ignore
import cv2 # type: ignore
import numpy as np # type: ignore
import os
from datetime import datetime

# Load and encode known faces
path = 'images'
known_encodings = []
known_names = []

print("[INFO] Loading known images...")

for person_name in os.listdir(path):
    person_folder = os.path.join(path, person_name)
    for img_file in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_file)
        img = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(person_name)

print("[INFO] Known faces loaded.")

# Mark attendance
def mark_attendance(name):
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d %H:%M:%S')

    if not os.path.exists('attendance.csv'):
        with open('attendance.csv', 'w') as f:
            f.write('Name,Time\n')

    with open('attendance.csv', 'r+') as f:
        lines = f.readlines()
        names_recorded = [line.split(',')[0] for line in lines]
        if name not in names_recorded:
            f.write(f'{name},{dt_string}\n')
            print(f"[ATTENDANCE] {name} marked at {dt_string}")

# Start webcam
cap = cv2.VideoCapture(0)

print("[INFO] Starting camera... Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for encoding, location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, encoding)
        face_distances = face_recognition.face_distance(known_encodings, encoding)

        if True in matches:
            best_match_index = np.argmin(face_distances)
            name = known_names[best_match_index]

            # Mark attendance
            mark_attendance(name)

            # Draw box
            y1, x2, y2, x1 = [val * 4 for val in location]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Camera stopped. Attendance session ended.")