import cv2

class CameraStream:
    def __init__(self, source):
        """
        source: 
          - int (webcam id)
          - str (rtsp url or video file path)
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

    def read(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def release(self):
        self.cap.release()
