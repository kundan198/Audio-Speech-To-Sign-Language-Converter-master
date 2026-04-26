import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import uuid
from pathlib import Path
import os
import numpy as np
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = ROOT / "assets"
DATA_DIR = ROOT / "data" / "sign_samples"
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"

def download_model():
    if not MODEL_PATH.parent.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print("Downloading MediaPipe hand_landmarker.task...")
        url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        urllib.request.urlretrieve(url, str(MODEL_PATH))

def process_video(video_path, landmarker):
    cap = cv2.VideoCapture(str(video_path))
    frames_landmarks = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    frame_idx = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        timestamp_ms = int((frame_idx / fps) * 1000)
        # Avoid timestamp 0 being repeated if fps calculation is tricky
        if frame_idx == 0: timestamp_ms = 1
        else: timestamp_ms = max(timestamp_ms, frame_idx * 33)
        
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        frame_data = []
        if result and result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                pts = [{"x": pt.x, "y": pt.y, "z": pt.z} for pt in hand_landmarks]
                frame_data.append(pts)
                
        frames_landmarks.append(frame_data)
        frame_idx += 1
        
    cap.release()
    return frames_landmarks

def augment_landmarks(landmarks, shift_x=0.0, shift_y=0.0, scale=1.0, noise=0.0):
    """
    landmarks: list of frames, each frame is list of hands, each hand is list of 21 points
    """
    aug_frames = []
    for frame in landmarks:
        aug_hands = []
        for hand in frame:
            aug_pts = []
            for pt in hand:
                nx = pt["x"] * scale + shift_x + np.random.normal(0, noise)
                ny = pt["y"] * scale + shift_y + np.random.normal(0, noise)
                nz = pt["z"] * scale + np.random.normal(0, noise)
                aug_pts.append({"x": nx, "y": ny, "z": nz})
            aug_hands.append(aug_pts)
        aug_frames.append(aug_hands)
    return aug_frames

def main():
    download_model()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    mp4_files = list(ASSETS_DIR.glob("*.mp4"))
    print(f"Found {len(mp4_files)} videos.")
    
    for idx, video_path in enumerate(mp4_files):
        label = video_path.stem
        print(f"[{idx+1}/{len(mp4_files)}] Processing {label}...")
        
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            landmarks = process_video(video_path, landmarker)
        
        # Check if hand was ever detected
        has_hands = any(len(frame) > 0 for frame in landmarks)
        if not has_hands:
            print(f"  -> No hands detected in {label}. Skipping.")
            continue
            
        label_dir = DATA_DIR / label
        label_dir.mkdir(exist_ok=True)
        
        # Save original
        orig_path = label_dir / f"{uuid.uuid4().hex}.json"
        with open(orig_path, 'w') as f:
            json.dump({"label": label, "landmarks": landmarks, "user": "system"}, f)
            
        # Generate 9 augmented versions
        for i in range(9):
            shift_x = np.random.uniform(-0.05, 0.05)
            shift_y = np.random.uniform(-0.05, 0.05)
            scale = np.random.uniform(0.9, 1.1)
            noise = np.random.uniform(0.001, 0.005)
            
            aug_landmarks = augment_landmarks(landmarks, shift_x, shift_y, scale, noise)
            aug_path = label_dir / f"{uuid.uuid4().hex}.json"
            with open(aug_path, 'w') as f:
                json.dump({"label": label, "landmarks": aug_landmarks, "user": "system"}, f)

if __name__ == "__main__":
    main()
