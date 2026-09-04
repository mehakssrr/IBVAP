from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.5):
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        """
        Returns list of dicts:
        [
          {"class": class_name, "bbox": [x1, y1, x2, y2], "conf": confidence},
          ...
        ]
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        detections = []
        r = results[0]
        boxes = r.boxes
        if boxes is None:
            return detections

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.model.names[cls_id]
            detections.append({
                "class": class_name,
                "bbox": [x1, y1, x2, y2],
                "conf": conf
            })
        return detections
