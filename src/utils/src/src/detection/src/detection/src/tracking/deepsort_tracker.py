import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

def iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter
    if union == 0:
        return 0.0
    return inter / union

class Track:
    def __init__(self, track_id, bbox, class_name):
        self.track_id = track_id
        self.class_name = class_name
        self.age = 1
        self.time_since_update = 0
        # Simple constant velocity model in x,y,w,h
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0,0],
            [0,1,0,0,0,1,0,0],
            [0,0,1,0,0,0,1,0],
            [0,0,0,1,0,0,0,1],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1],
        ])
        self.kf.H = np.eye(4, 8)
        self.kf.R *= 10.0
        self.kf.P *= 1000.0
        self.kf.Q = Q_discrete_white_noise(4, dt=1.0, var=1.0)

        x, y, w, h = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]
        self.kf.x[:4] = np.array([x, y, w, h, 0, 0, 0, 0]).reshape(8, 1)

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox):
        x, y, w, h = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]
        self.kf.update(np.array([x, y, w, h]).reshape(4, 1))
        self.time_since_update = 0

    def get_state(self):
        x, y, w, h = self.kf.x[:4].flatten()
        return [int(x), int(y), int(x+w), int(y+h)]

class DeepSORTTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1

    def update(self, detections, frame=None):
        # detections: list of {"class": ..., "bbox": [x1,y1,x2,y2], "conf": ...}
        if len(detections) == 0:
            # No detections; age all tracks
            for tr in self.tracks:
                tr.predict()
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return [t for t in self.tracks if t.age >= self.min_hits]

        # Associate detections to existing tracks using IoU
        matched = [False] * len(detections)
        unmatched_tracks = list(range(len(self.tracks)))

        for ti, tr in enumerate(self.tracks):
            tr_bbox = tr.get_state()
            best_iou = -1
            best_di = -1
            for di, det in enumerate(detections):
                if matched[di]:
                    continue
                iou_val = iou(tr_bbox, det["bbox"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_di = di
            if best_iou >= self.iou_threshold:
                matched[best_di] = True
                unmatched_tracks.remove(ti)
                tr.update(detections[best_di]["bbox"])

        # Create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if not matched[di]:
                tr = Track(self.next_id, det["bbox"], det["class"])
                self.next_id += 1
                self.tracks.append(tr)

        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Predict for next frame
        for tr in self.tracks:
            tr.predict()

        # Return only "confirmed" tracks
        return [t for t in self.tracks if t.age >= self.min_hits]
