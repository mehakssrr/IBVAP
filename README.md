# IBVAP – Intelligent Border Video Analytics Platform

A software-defined video analytics platform that turns standard CCTV/IP cameras into an intelligent surveillance system using AI.

## Features (current)

- Real-time human/vehicle detection using YOLOv8  
- Multi-object tracking with a Kalman+IoU tracker  
- Virtual fence intrusion detection  
- Event logging (text log)  
- Debug visualization window

## Setup

```bash
pip install -r requirements.txt
```

Place a video file at `data/sample.mp4` or edit `configs/default.yaml` to use your RTSP URL.

## Run

```bash
python -m src.main
```

Press `q` in the video window to quit.

## Next steps

- Add ANPR (license plate detection + OCR)  
- Add face detection/recognition  
- Replace simple tracker with DeepSORT/ByteTrack  
- Add REST API / dashboard for alerts  
- Add privacy-preserving anonymization at the edge
