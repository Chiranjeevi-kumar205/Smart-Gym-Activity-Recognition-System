import cv2
import numpy as np

def draw_annotations(frame, tracks):
    """
    Draws bounding boxes and track information on the frame.
    """
    for track_id, data in tracks.items():
        box = data['box']
        activity = data.get('last_activity', 'N/A')
        
        x1, y1, x2, y2 = map(int, box)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepare text labels
        id_text = f"ID: {track_id}"
        activity_text = f"Activity: {activity}"
        
        # Draw background for text
        cv2.rectangle(frame, (x1, y1 - 45), (x1 + 180, y1), (0, 255, 0), -1)
        
        # Put text on the frame
        cv2.putText(frame, id_text, (x1 + 5, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, activity_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
    return frame

def read_video_frames(video_path):
    """
    Helper function to read all frames from a video file.
    Note: Not used in the main real-time loop, but useful for classifier testing.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames