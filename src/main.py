from src.stream_ingest import CameraStream
from src.detection.yolo_detector import YOLODetector
from src.tracking.deepsort_tracker import DeepSORTTracker
from src.analytics.virtual_fence import VirtualFence
from src.alerts.alert_manager import AlertManager

def main():
    stream = CameraStream(rtsp_url="rtsp://...")
    detector = YOLODetector(model="yolov8n.pt")
    tracker = DeepSORTTracker()
    fence = VirtualFence(polygon=[[100,200], [300,200], [300,400], [100,400]])
    alerter = AlertManager(webhook="http://localhost:8000/alert")

    for frame in stream.read():
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)
        
        for track in tracks:
            if fence.is_breach(track.bbox):
                alerter.send_alert("INTRUSION", track)

        # Optional: display for debugging
        # cv2.imshow("IBVAP", frame); cv2.waitKey(1)

if __name__ == "__main__":
    main()
