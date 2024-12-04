import cv2
import numpy as np
import mediapipe as mp
import dlib
from math import hypot
from playsound import playsound
import threading  
from ultralytics import YOLO  

yolo_model = YOLO("best.pt")

cap = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

frames_with_closed_eyes = 0
threshold_frames = 5
frames_leaning_forward = 0
leaning_threshold_frames = 10
blinking_ratio = 0 
audio_path = "alarm.wav" 
class_count = 0
alert_triggered = False

def play_alert_sound():
    playsound(audio_path)

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def midpoint2(p1, p2):
    return float((p1[0] + p2[0]) / 2)

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_length / ver_line_length
    return ratio

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    yolo_results = yolo_model.predict(frame, stream=True, conf=0.35)

    for result in yolo_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            confidence = box.conf[0]  
            class_name = yolo_model.names[int(box.cls[0])].lower()

            if class_name == 'drowsy':  
                class_count += 1
                cv2.putText(frame, f"ALERTA: sonolencia - Score{class_count}", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if class_count >= 15 and not alert_triggered:
                    threading.Thread(target=play_alert_sound, daemon=True).start()
                    alert_triggered = True
            else:
                class_count = 0
                alert_triggered = False

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    for face in faces:
        landmarks = predictor(gray, face)
        
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)

        left_eye_points = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(36, 42)])
        right_eye_points = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(42, 48)])
        
        left_eye_hull = cv2.convexHull(left_eye_points)
        right_eye_hull = cv2.convexHull(right_eye_points)

        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        if blinking_ratio > 4.5:
            frames_with_closed_eyes += 1
        else:
            frames_with_closed_eyes = 0

        if frames_with_closed_eyes >= threshold_frames:
            cv2.putText(frame, f"ALERTA: Olhos fechados - Score{frames_with_closed_eyes}", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if frames_with_closed_eyes == 10:  
                threading.Thread(target=play_alert_sound, daemon=True).start()

    if results.pose_landmarks:
        nose = results.pose_landmarks.landmark[0]
        left_shoulder = results.pose_landmarks.landmark[11]
        right_shoulder = results.pose_landmarks.landmark[12]
        
        nose_point = np.array([nose.x, nose.y, nose.z])
        left_shoulder_point = np.array([left_shoulder.x, left_shoulder.y, left_shoulder.z])
        right_shoulder_point = np.array([right_shoulder.x, right_shoulder.y, right_shoulder.z])

        midpoint_shoulder = midpoint2(left_shoulder_point, right_shoulder_point)
        nose_to_left_shoulder_dist = np.linalg.norm(nose_point - left_shoulder_point)
        nose_to_right_shoulder_dist = np.linalg.norm(nose_point - right_shoulder_point)

        sholder_to_nose_dist = np.linalg.norm(nose_point - midpoint_shoulder)

        if sholder_to_nose_dist > 2.9:
            frames_leaning_forward += 1
        else:
            frames_leaning_forward = 0

        if frames_leaning_forward >= leaning_threshold_frames:
            cv2.putText(frame, f"ALERTA: Inclinacao - Score{frames_leaning_forward}", 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if frames_leaning_forward == 10:  
                threading.Thread(target=play_alert_sound, daemon=True).start()

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
