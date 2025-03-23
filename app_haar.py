import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import gradio as gr

model = load_model('Honours-TY-Proj/models/best_MoBiLSTM_model.keras')

haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

frame_height, frame_width = 96, 96
sequence_length = 16

def preprocess_frames(frames):
    processed = [cv2.resize(frame, (frame_width, frame_height)).astype(
        'float32') / 255.0 for frame in frames]
    return np.expand_dims(processed, axis=0)

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def detect_with_haar(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = haar_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.05,  
        minNeighbors=8,    
        minSize=(40, 40),  
        maxSize=(200, 200) 
    )

    filtered_detections = []
    for i, (x, y, w, h) in enumerate(detections):
        overlap = False
        for j, (x2, y2, w2, h2) in enumerate(filtered_detections):
            if (x < x2 + w2 and x + w > x2 and y < y2 + h2 and y + h > y2):
                overlap = True
                break
        if not overlap:
            filtered_detections.append((x, y, w, h))

    for (x, y, w, h) in filtered_detections:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    print(f"Haar Cascade Detections: {len(filtered_detections)}")

    return frame, len(filtered_detections) > 0

def classify_frames(video_path):
    frames = extract_frames(video_path)
    labeled_frames = []

    if len(frames) < sequence_length:
        frames += [frames[-1]] * (sequence_length - len(frames))

    sequences = [
        frames[i:i + sequence_length]
        for i in range(0, len(frames) - sequence_length + 1, sequence_length)
    ]

    processed_sequences = np.vstack([preprocess_frames(seq) for seq in sequences])

    predictions = model.predict(processed_sequences)

    for seq_idx, frame_sequence in enumerate(sequences):
        label = "Violence" if np.argmax(predictions[seq_idx]) == 1 else "Non-Violence"
        color = (0, 0, 255) if label == "Violence" else (0, 255, 0)

        for frame in frame_sequence:
            annotated_frame = frame.copy()
            annotated_frame, detected = detect_with_haar(annotated_frame)
            haar_label = "Face Detected" if detected else "No Face Detected"

            cv2.putText(annotated_frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(annotated_frame, haar_label, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            labeled_frames.append(annotated_frame)

    return labeled_frames

def process_and_display(video_file):
    labeled_frames = classify_frames(video_file)
    video_path = "output.mp4"
    height, width, layers = labeled_frames[0].shape
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(
        *'mp4v'), 20, (width, height))

    for frame in labeled_frames:
        out.write(frame)
    out.release()

    return video_path


interface = gr.Interface(
    fn=process_and_display,
    inputs=gr.Video(label="Upload a Video"),
    outputs=gr.Video(label="Labeled Video Output"),
    title="Violence Detection System with Haar Cascade",
    description="Upload a video to detect and label as containing violent or non-violent scenes. Includes Haar Cascade detection."
)

if __name__ == "__main__":
    interface.launch()