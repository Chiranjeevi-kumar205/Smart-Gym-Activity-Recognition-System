import numpy as np
from scipy.optimize import linear_sum_assignment
import config

def iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou_val = interArea / float(boxAArea + boxBArea - interArea)
    return iou_val

class Tracker:
    """
    A simple tracker that uses IoU and the Hungarian algorithm for assignment.
    """
    def __init__(self):
        self.tracks = {}
        self.next_track_id = 0

    def update(self, detections):
        """
        Update tracks with new detections.
        """
        # If no tracks, initialize with new detections
        if not self.tracks:
            for det in detections:
                self.tracks[self.next_track_id] = {'box': det['box'], 'frames': [], 'last_activity': 'N/A'}
                self.next_track_id += 1
            return

        # Prepare cost matrix for assignment
        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid]['box'] for tid in track_ids]
        det_boxes = [det['box'] for det in detections]

        cost_matrix = np.zeros((len(track_boxes), len(det_boxes)))
        for i, track_box in enumerate(track_boxes):
            for j, det_box in enumerate(det_boxes):
                cost_matrix[i, j] = 1 - iou(track_box, det_box)

        # Use Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_track_ids = set()
        matched_det_indices = set(col_ind)

        # Update matched tracks
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < (1 - config.IOU_THRESHOLD):
                track_id = track_ids[r]
                self.tracks[track_id]['box'] = det_boxes[c]
                matched_track_ids.add(track_id)

        # Add new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_det_indices:
                self.tracks[self.next_track_id] = {'box': det['box'], 'frames': [], 'last_activity': 'N/A'}
                self.next_track_id += 1

        # Note: A real-world tracker would also handle track disappearance/reappearance.
        # This simple version focuses on continuous tracking.