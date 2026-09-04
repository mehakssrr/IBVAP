import cv2
import numpy as np

class VirtualFence:
    def __init__(self, polygon, enabled=True):
        """
        polygon: list of [x, y]
        """
        self.enabled = enabled
        self.polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))

    def is_inside(self, point):
        if not self.enabled:
            return False
        x, y = point
        pt = np.array([[x, y]], dtype=np.float32)
        return cv2.pointPolygonTest(self.polygon, (x, y), False) >= 0

    def draw_on_frame(self, frame, color=(0, 255, 0), thickness=2):
        if not self.enabled:
            return frame
        return cv2.polylines(frame, [self.polygon], isClosed=True, color=color, thickness=thickness)
