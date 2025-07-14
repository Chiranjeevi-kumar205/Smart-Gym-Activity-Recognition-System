from ultralytics import YOLO
import config

class PersonDetector:
    """
    A class to detect persons in a video frame using YOLOv8.
    """
    def __init__(self, model_path=config.YOLO_MODEL_PATH):
        """
        Initializes the PersonDetector with a YOLOv8 model.
        
        Args:
            model_path (str): The path to the YOLOv8 model file.
        """
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Performs person detection on a single frame.

        Args:
            frame (np.ndarray): The input video frame.

        Returns:
            list: A list of detection results, where each result contains
                  the bounding box coordinates and confidence score.
        """
        # Perform inference
        results = self.model.predict(frame, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                # Check if the detected object is a person and meets the confidence threshold
                if int(box.cls) == config.PERSON_CLASS_ID and box.conf >= config.CONFIDENCE_THRESHOLD:
                    # Bounding box coordinates (x1, y1, x2, y2)
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    detections.append({'box': xyxy, 'confidence': confidence})
        
        return detections