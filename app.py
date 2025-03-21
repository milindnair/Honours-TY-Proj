import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import gradio as gr

# Load the model
model = load_model('mobilstm_model.keras')

# Frame dimensions and sequence length
frame_height, frame_width = 96, 96
sequence_length = 16

# Preprocess frames for model input
def preprocess_frames(frames):
    processed = [cv2.resize(frame, (frame_width, frame_height)).astype(
        'float32') / 255.0 for frame in frames]
    return np.expand_dims(processed, axis=0)

# Extract frames from video
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

# Classify frames in sequences of 16
def classify_frames(video_path):
    frames = extract_frames(video_path)
    labeled_frames = []

    # Pad frames if less than sequence_length
    if len(frames) < sequence_length:
        frames += [frames[-1]] * (sequence_length - len(frames))

    for i in range(0, len(frames) - sequence_length + 1, sequence_length):
        frame_sequence = frames[i:i + sequence_length]
        processed_frames = preprocess_frames(frame_sequence)
        prediction = model.predict(processed_frames)[0]
        label = "Violence" if np.argmax(prediction) == 1 else "Non-Violence"

        for frame in frame_sequence:
            annotated_frame = frame.copy()
            color = (0, 0, 255) if label == "Violence" else (0, 255, 0)
            cv2.putText(annotated_frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            labeled_frames.append(annotated_frame)

    return labeled_frames

# Gradio interface
def process_and_display(video_file):
    labeled_frames = classify_frames(video_file)

    # Convert labeled frames to a format compatible with Gradio
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
    title="Violence Detection System",
    description="Upload a video to detect and label as containing violent or non-violent scenes."
)

if __name__ == "__main__":
    interface.launch()
