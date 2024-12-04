import cv2
import numpy as np
import mediapipe as mp
import dlib
from math import hypot
from playsound import playsound
import threading
import streamlit as st
from ultralytics import YOLO

yolo_model = YOLO("best.pt")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def play_alert_sound():
    playsound("alarm.wav")

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
    return hor_line_length / ver_line_length

def update_alerts(score_closed_eyes, score_leaning, score_drowsy, alert_triggered):
    alert_message = ""
    if score_closed_eyes >= 5:
        alert_message += f"ALERTA: Olhos fechados - Score {score_closed_eyes}\n"
        if score_closed_eyes == 5 and not alert_triggered:
            threading.Thread(target=play_alert_sound, daemon=True).start()
            alert_triggered = True
    if score_leaning >= 10:
        alert_message += f"ALERTA: Inclinacao - Score {score_leaning}\n"
        if score_leaning == 10 and not alert_triggered:
            threading.Thread(target=play_alert_sound, daemon=True).start()
            alert_triggered = True
    if score_drowsy >= 7:
        alert_message += f"ALERTA: Sonolencia - Score {score_drowsy}\n"
        if score_drowsy == 15 and not alert_triggered:
            threading.Thread(target=play_alert_sound, daemon=True).start()
            alert_triggered = True
    return alert_message, alert_triggered

st.title("Sistema de Deteccao de Sonolencia")
enable_video = st.checkbox("Habilitar camera", value=True)

def video_stream():
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    alert_placeholder = st.empty()
    prev_time = 0

    frames_with_closed_eyes = 0
    threshold_frames = 5
    frames_leaning_forward = 0
    leaning_threshold_frames = 10
    class_count = 0
    alert_triggered = False

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Recarregue novamente o site para acessar os dados da camera")
            break

        current_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (current_time - prev_time)
        prev_time = current_time

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        yolo_results = yolo_model.predict(frame, stream=True, conf=0.35, imgsz=640)
        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf[0]
                class_name = yolo_model.names[int(box.cls[0])].lower()
                if class_name == 'drowsy':
                    class_count += 1
                else:
                    class_count = 0

        for face in faces:
            landmarks = predictor(gray, face)
            left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
            right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

            if blinking_ratio > 4.5:
                frames_with_closed_eyes += 1
            else:
                frames_with_closed_eyes = 0

        if results.pose_landmarks:
            nose = results.pose_landmarks.landmark[0]
            left_shoulder = results.pose_landmarks.landmark[11]
            right_shoulder = results.pose_landmarks.landmark[12]
            nose_point = np.array([nose.x, nose.y, nose.z])
            left_shoulder_point = np.array([left_shoulder.x, left_shoulder.y, left_shoulder.z])
            right_shoulder_point = np.array([right_shoulder.x, right_shoulder.y, right_shoulder.z])
            midpoint_shoulder = midpoint2(left_shoulder_point, right_shoulder_point)
            sholder_to_nose_dist = np.linalg.norm(nose_point - midpoint_shoulder)

            if sholder_to_nose_dist > 2.9:
                frames_leaning_forward += 1
            else:
                frames_leaning_forward = 0

        alert_message, alert_triggered = update_alerts(
            frames_with_closed_eyes,
            frames_leaning_forward,
            class_count,
            alert_triggered
        )

        alert_placeholder.text(alert_message)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print('fps',fps)
        if enable_video:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB")

    cap.release()



video_stream()
