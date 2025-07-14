import torch

# --- System Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Video Source ---
# Use 0 for webcam or provide a path to a video file
VIDEO_SOURCE = 'videos/gym_session.mp4' 

# --- Model Configuration ---
# Detection Model
YOLO_MODEL_PATH = 'yolov8n.pt'  # Nano version for speed, can be yolov8s.pt, yolov8m.pt etc.

# Classification Model
# Using a smaller, more general model. Replace with a fine-tuned one if available.
TIMESFORMER_MODEL_PATH = 'facebook/timesformer-base-finetuned-k400'

# --- Detection & Tracking Parameters ---
CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for YOLOv8
IOU_THRESHOLD = 0.4         # IOU threshold for tracker matching
PERSON_CLASS_ID = 0         # In COCO dataset, 'person' is class 0

# --- Activity Classification Parameters ---
# Number of frames to collect for each person before running classification
CLIP_BUFFER_SIZE = 32
# Activities to recognize (must match labels from the TimeSformer model's config)
# Find labels here: https://huggingface.co/facebook/timesformer-base-finetuned-k400/blob/main/config.json
TARGET_ACTIVITIES = {
    'squat', 
    'push up', 
    'pull up', 
    'biceps curl', 
    'bench press',
    'cycling', 
    'rowing machine'
}

# --- Output Configuration ---
OUTPUT_LOG_FILE = 'activity_log.csv'
OUTPUT_VIDEO_FILE = 'output.mp4'