import face_recognition
import cv2
import os
import json

# Load user data
with open("user_data.json", "r") as f:
    user_data = json.load(f)

# Load known faces
known_face_encodings = []
known_face_names = []

for filename in os.listdir("known_faces"):
    if filename.endswith(".jpg"):
        image = face_recognition.load_image_file(f"known_faces/{filename}")
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_face_encodings.append(encoding[0])
            name_key = filename.split(".")[0]
            known_face_names.append(name_key)

# --- Function to pick correct webcam ---
def get_webcam_index(max_index=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print(f"Working camera found at index {i}")
                return i
    print("No working camera found, defaulting to 0")
    return 0

camera_index = get_webcam_index()
video_capture = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame from camera")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name_key = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name_key = known_face_names[first_match_index]
            user_info = user_data.get(name_key, {})
            display_text = f"{user_info.get('name', 'Unknown')} | Age: {user_info.get('age', 'N/A')} | Height: {user_info.get('height', 'N/A')}"
        else:
            display_text = "Unknown"

        # Scale back up face locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, display_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Face Recognition App", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
