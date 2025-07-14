import torch
from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import config
from utils.video_utils import read_video_frames

class ActivityClassifier:
    """
    A class to classify activities from video clips using TimeSformer.
    """
    def __init__(self, model_path=config.TIMESFORMER_MODEL_PATH):
        """
        Initializes the classifier with a pre-trained TimeSformer model.
        """
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = TimesformerForVideoClassification.from_pretrained(model_path)
        self.model.to(config.DEVICE)
        self.model.eval()

    def classify(self, video_frames):
        """
        Classifies the activity in a list of video frames.

        Args:
            video_frames (list): A list of video frames (as numpy arrays).

        Returns:
            str: The predicted activity label, or None if no target activity is detected.
        """
        if not video_frames:
            return None

        # The model expects a certain number of frames, let's sample uniformly
        num_frames = self.model.config.num_frames
        indices = np.linspace(0, len(video_frames) - 1, num=num_frames, dtype=int)
        sampled_frames = [video_frames[i] for i in indices]

        # Preprocess the frames
        inputs = self.processor(images=sampled_frames, return_tensors="pt")
        inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get the top prediction
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = self.model.config.id2label[predicted_class_idx]
        
        print(f"TimeSformer raw prediction: {predicted_label}")

        # Check if the prediction is one of our target activities
        if predicted_label in config.TARGET_ACTIVITIES:
            return predicted_label
        
        return None