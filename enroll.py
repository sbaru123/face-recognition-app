"""
enroll.py — Student enrollment via ID card scan

Usage:
    python enroll.py

Flow:
    1. Hold student ID card up to the webcam so it fills the frame
    2. Press SPACE to capture
    3. App detects the face on the card and OCRs the name + student ID
    4. Confirm the details, then press ENTER to save or R to retry
"""

import cv2
import face_recognition
import easyocr
import mediapipe as mp
import re
import numpy as np
from database import save_student

# --- Setup ---
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.4)
ocr_reader = easyocr.Reader(["en"], gpu=False)


def get_webcam_index(max_index=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                return i
    return 0


def detect_face_on_card(frame):
    """Detect a face in the frame (the ID card photo) and return its encoding."""
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)

    if not results.detections:
        return None, None

    # Pick the detection with the highest confidence
    best = max(results.detections, key=lambda d: d.score[0])
    bbox = best.location_data.relative_bounding_box

    left   = max(0, int(bbox.xmin * w))
    top    = max(0, int(bbox.ymin * h))
    right  = min(w, int((bbox.xmin + bbox.width) * w))
    bottom = min(h, int((bbox.ymin + bbox.height) * h))

    face_loc = [(top, right, bottom, left)]
    encodings = face_recognition.face_encodings(rgb, face_loc)

    if not encodings:
        return None, None

    return encodings[0], (top, right, bottom, left)


def ocr_card(frame):
    """Run OCR on the frame and extract name + grade level.
    Tries normal orientation first, then rotations to catch vertical grade text."""
    # Try standard orientation + 90/270 degree rotations for vertical text
    all_texts = []
    for result in ocr_reader.readtext(frame, rotation_info=[90, 180, 270]):
        all_texts.append(result[1])

    full_text = " ".join(all_texts)
    print(f"  Raw OCR text: {full_text}")

    # Grade heuristic — matches "Grade 10", "Gr 9", "10th", "Grade: 11", plain "9" etc.
    grade_match = re.search(
        r"\b(?:grade\s*:?\s*)?(\d{1,2})(?:st|nd|rd|th)?\b|\b(?:gr\.?\s*)(\d{1,2})\b",
        full_text, re.IGNORECASE
    )
    if grade_match:
        grade = grade_match.group(1) or grade_match.group(2)
    else:
        grade = None

    # Name heuristic: longest alphabetic token (typical on IDs)
    name_candidates = [t for t in all_texts if len(t) > 3 and t.replace(" ", "").isalpha()]
    name = max(name_candidates, key=len) if name_candidates else None

    return name, grade, full_text


def main():
    cam_index = get_webcam_index()
    cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)

    # Warm up
    for _ in range(10):
        cap.read()

    print("\n=== Enrollment Mode ===")
    print("Hold the student ID card up to the camera.")
    print("Press SPACE to capture | Press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error")
            break

        display = frame.copy()
        cv2.putText(display, "Hold ID up to camera. Press SPACE to capture.",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow("Enrollment", display)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" "):
            print("Capturing...")

            # Step 1: Detect face on ID card
            encoding, face_loc = detect_face_on_card(frame)
            if encoding is None:
                print("No face detected on card — try holding it closer or flatter.")
                continue

            # Draw the detected face box
            top, right, bottom, left = face_loc
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Step 2: OCR name + grade
            print("Running OCR...")
            name, grade, raw_text = ocr_card(frame)
            print(f"  Detected name: {name}")
            print(f"  Detected grade: {grade}")

            # Step 3: Let user confirm or correct
            if not name:
                name = input("  Could not read name — enter manually: ").strip()
            else:
                override = input(f"  Name detected as '{name}'. Press ENTER to confirm or type correction: ").strip()
                if override:
                    name = override

            if not grade:
                grade = input("  Could not read grade — enter manually (e.g. 10): ").strip()
            else:
                override = input(f"  Grade detected as '{grade}'. Press ENTER to confirm or type correction: ").strip()
                if override:
                    grade = override

            # Step 4: Save to database
            student = save_student(name, grade, encoding)
            if student:
                print(f"\nSuccessfully enrolled: {name} (Grade {grade})")
                print("Press SPACE to enroll another student, or Q to quit.\n")

    cap.release()
    face_detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
