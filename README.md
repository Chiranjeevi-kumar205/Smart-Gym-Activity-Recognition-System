# Smart Gym Activity Recognition System

This project is an end-to-end pipeline for detecting individuals in a gym, tracking their movements, and classifying their physical activities using state-of-the-art deep learning models. The system logs the performed activities for each person, providing a foundation for fitness analytics and progress tracking.

## Features

- **Person Detection & Tracking**: Utilizes **YOLOv8** to detect and track multiple individuals in video streams with high accuracy.
- **Dynamic Clip Generation**: Automatically generates short video clips for each tracked person, maintaining their identity across frames.
- **Spatio-Temporal Activity Classification**: Employs a pre-trained **TimeSformer** model to classify complex physical activities (e.g., squats, push-ups, cycling) from the generated clips.
- **Automated Data Logging**: Logs all classified activities with a timestamp and a unique person ID to a CSV file for further analysis.
- **Offline & Modular Design**: The system is built to run entirely offline. Its modular components (detection, tracking, classification) can be easily updated or integrated into other applications.

## Tech Stack

- **Computer Vision**: YOLOv8, OpenCV
- **Deep Learning**: PyTorch, HuggingFace Transformers
- **Models**: YOLOv8 (for detection), TimeSformer (for classification)

## Directory Structure