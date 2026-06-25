# Human Detection and Target Engagement Simulation using Computer Vision

## DRDO Computer Vision Internship Project

---

## Overview

This repository contains the implementation of a modular Computer Vision system developed during my internship at the **Defence Research and Development Organisation (DRDO)**. The project focuses on real-time human detection, face detection, pose estimation, anatomical keypoint localization, multi-target tracking, and a simulated target engagement framework.

The implementation demonstrates the integration of modern deep learning models with real-time tracking, visualization, target analytics, and event logging in a modular Python application. The project emphasizes software architecture, computer vision, and intelligent target analysis while providing a research-oriented simulation environment.

---
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)


## Project Objectives

* Perform real-time human detection using YOLOv8.
* Detect human faces and estimate body pose.
* Estimate anatomical chest reference points using body proportions.
* Track multiple targets with persistent identities.
* Estimate target distance, bearing, and elevation.
* Implement a modular target engagement simulation framework.
* Record system events and session analytics.
* Develop a scalable and production-ready software architecture.

---

# Key Features

### Human Detection

Real-time human detection using the YOLOv8 object detection model.

### Face Detection

Dedicated face detection pipeline for enhanced localization and distance estimation.

### Human Pose Estimation

Extraction of body keypoints using the YOLOv8 Pose model.

### Anatomical Keypoint Estimation

Dynamic estimation of anatomical chest reference points using body proportions instead of fixed pixel offsets, providing greater robustness across varying target distances.

### Multi-Target Tracking

Persistent target identities (T-001, T-002, etc.) are maintained using an Intersection over Union (IoU) based tracking algorithm with motion prediction.

### Tactical HUD

A real-time visualization interface displaying:

* Bounding boxes
* Target IDs
* Confidence scores
* Estimated distance
* Bearing
* Elevation
* Lock status
* Motion trails
* Tactical minimap

### Target Engagement Simulation

A configurable simulation module supports:

* Safety state management
* Target lock progression
* Manual engagement simulation
* Automated engagement simulation

The implementation is intended for software validation and visualization within a research environment.

### Session Logging

The application automatically records:

* Timestamped event logs
* Target coordinates
* Estimated distance
* Lock events
* Session screenshots
* Video recordings

---

# Software Architecture

```
DRDO Computer Vision Project
│
├── main.py
├── config.py
├── requirements.txt
│
├── core/
│   ├── detector.py
│   ├── heart_estimator.py
│   ├── tracker.py
│   └── firing_system.py
│
├── ui/
│   ├── colors.py
│   └── hud.py
│
├── utils/
│   ├── geometry.py
│   └── logger.py
│
├── logs/
└── recordings/
```

---

# Technology Stack

* Python
* OpenCV
* Ultralytics YOLOv8
* NumPy

---

# Core Modules

| Module             | Description                                                              |
| ------------------ | ------------------------------------------------------------------------ |
| detector.py        | Multi-model detection engine combining person, face, and pose estimation |
| tracker.py         | Persistent multi-target tracking using IoU and velocity estimation       |
| heart_estimator.py | Anatomical reference point estimation using body proportions             |
| firing_system.py   | Target lock management and engagement simulation                         |
| hud.py             | Tactical visualization interface                                         |
| geometry.py        | Distance, bearing, smoothing, and geometric utilities                    |
| logger.py          | Event logging and recording management                                   |

---

# System Workflow

```
Camera / Video Input
          │
          ▼
YOLO Human Detection
          │
          ▼
Face Detection
          │
          ▼
Pose Estimation
          │
          ▼
Anatomical Keypoint Estimation
          │
          ▼
Target Tracking
          │
          ▼
Distance & Bearing Estimation
          │
          ▼
HUD Visualization
          │
          ▼
Target Engagement Simulation
          │
          ▼
Session Logging
```

---

# Keyboard Controls

| Key   | Function                               |
| ----- | -------------------------------------- |
| Q     | Exit Application                       |
| S     | Toggle Safety State                    |
| F     | Toggle Automated Engagement Simulation |
| SPACE | Manual Engagement Simulation           |
| G     | Toggle Tactical Grid                   |
| M     | Toggle Minimap                         |
| T     | Toggle Motion Trails                   |
| P     | Capture Screenshot                     |

---

# Running the Project

```bash
pip install -r requirements.txt
```

Default camera:

```bash
.\venv\Scripts\python.exe main.py
```

Video file:

```bash
.\venv\Scripts\python.exe main.py --source path/to/video.mp4 --record
```

Record session:

```bash
python main.py --record
```

---

# Skills Demonstrated

* Computer Vision
* Object Detection
* Human Pose Estimation
* Multi-Object Tracking
* Deep Learning
* OpenCV
* YOLOv8
* Software Architecture
* Real-Time Video Processing
* Modular Python Development
* Data Logging and Visualization

---

# Future Enhancements

* DeepSORT / ByteTrack Integration
* Kalman Filter Motion Prediction
* GPU Optimization
* Edge AI Deployment
* ONNX/TensorRT Inference
* Multi-Camera Support
* Performance Benchmarking

---

# Acknowledgement

This project was developed during my internship at the **Defence Research and Development Organisation (DRDO)** as part of a Computer Vision research assignment. The repository showcases the software engineering, computer vision, and deep learning components that are suitable for public presentation and does not include confidential or restricted information.

---

# Author

**Dhruv Chauhan**

B.Tech – Computer Science Engineering

Computer Vision • Deep Learning • Artificial Intelligence
