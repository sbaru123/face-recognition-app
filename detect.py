import cv2
import face_recognition
import mediapipe as mp
import numpy as np
from collections import deque
from database import load_all_encodings, log_attendance

# --- Load known encodings from DB ---
print("Loading face encodings from database...")
known_encodings, known_students = load_all_encodings()
print(f"Loaded {len(known_encodings)} encoding(s) for {len(set(s.id for s in known_students))} student(s).")

# --- MediaPipe setup ---
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


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

# Warm up camera
for _ in range(10):
    video_capture.read()

frame_count = 0
last_face_locations = []
last_display_texts   = []

# Rolling vote buffer — smooths flickering across last N detections
SMOOTHING_FRAMES = 8
name_history = deque(maxlen=SMOOTHING_FRAMES)

# Track which students have already been logged this session (avoid duplicates)
logged_today = set()

print("Starting attendance detection. Press Q to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame from camera")
        break

    frame_count += 1

    if frame_count % 3 == 0:
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detector.process(rgb_frame)

        last_face_locations = []
        last_display_texts   = []

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                left   = max(0, int(bbox.xmin * w))
                top    = max(0, int(bbox.ymin * h))
                right  = min(w, int((bbox.xmin + bbox.width) * w))
                bottom = min(h, int((bbox.ymin + bbox.height) * h))

                face_loc  = [(top, right, bottom, left)]
                encodings = face_recognition.face_encodings(rgb_frame, face_loc)

                display_text = "Unknown"

                if encodings and known_encodings:
                    distances = face_recognition.face_distance(known_encodings, encodings[0])
                    best_idx  = int(np.argmin(distances))

                    if distances[best_idx] < 0.6:
                        matched_student = known_students[best_idx]
                        name_history.append(matched_student.id)
                    else:
                        name_history.append(None)

                    # Majority vote over rolling buffer
                    known_votes = [n for n in name_history if n is not None]
                    if known_votes:
                        best_id = max(set(known_votes), key=known_votes.count)
                        # Find the student object
                        student = next((s for s in known_students if s.id == best_id), None)
                        if student:
                            display_text = f"{student.name} | Grade {student.grade}"

                            # Log attendance if not already logged this session
                            if student.id not in logged_today:
                                logged = log_attendance(student.id)
                                if logged:
                                    print(f"Attendance logged: {student.name} ({student.grade})")
                                logged_today.add(student.id)

                last_face_locations.append((top, right, bottom, left))
                last_display_texts.append(display_text)

    # Draw on every frame using last known positions
    for (top, right, bottom, left), display_text in zip(last_face_locations, last_display_texts):
        color = (0, 255, 0) if display_text != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, display_text, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Attendance — Press Q to quit", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

video_capture.release()
face_detector.close()
cv2.destroyAllWindows()
print(f"\nSession ended. Students logged today: {len(logged_today)}")
