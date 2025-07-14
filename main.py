import cv2
import config
from modules.person_detector import PersonDetector
from modules.activity_classifier import ActivityClassifier
from modules.data_logger import DataLogger
from modules.tracker import Tracker
from utils.video_utils import draw_annotations
import time

def main():
    # --- Initialization ---
    detector = PersonDetector()
    tracker = Tracker()
    classifier = ActivityClassifier()
    logger = DataLogger(config.OUTPUT_LOG_FILE)

    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source {config.VIDEO_SOURCE}")
        return

    # Video writer setup
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(config.OUTPUT_VIDEO_FILE, fourcc, fps, (frame_width, frame_height))

    print("Processing video... Press 'q' to quit.")
    
    # --- Main Processing Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detect persons in the current frame
        detections = detector.detect(frame)

        # 2. Update tracker with new detections
        tracker.update(detections)

        # 3. Process each tracked person
        for track_id, track_data in list(tracker.tracks.items()):
            # Crop the person from the frame to store in the buffer
            x1, y1, x2, y2 = track_data['box']
            person_crop = frame[y1:y2, x1:x2]
            
            # Append the cropped frame to the buffer for this person
            track_data['frames'].append(person_crop)

            # 4. If buffer is full, classify activity
            if len(track_data['frames']) >= config.CLIP_BUFFER_SIZE:
                print(f"Buffer full for Person ID {track_id}. Classifying activity...")
                
                # Get the buffered frames for classification
                clip_frames = track_data['frames']
                
                # Classify the activity
                activity = classifier.classify(clip_frames)

                if activity:
                    # 5. Log activity and update track state
                    logger.log(track_id, activity)
                    track_data['last_activity'] = activity
                
                # 6. Clear the buffer to start collecting for the next action
                track_data['frames'].clear()
        
        # 7. Draw annotations on the frame
        annotated_frame = draw_annotations(frame.copy(), tracker.tracks)
        
        # 8. Display and save the frame
        cv2.imshow('Smart Gym System', annotated_frame)
        out.write(annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing finished. Output video and log file saved.")


if __name__ == "__main__":
    main()