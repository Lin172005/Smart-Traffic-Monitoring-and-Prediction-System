# Smart Traffic Monitoring and Prediction System

## Overview
The **Smart Traffic Monitoring and Prediction System** is a Python-based project that analyzes traffic conditions using pre-recorded video footage. It employs YOLOv8 for vehicle detection, calculates traffic density, predicts congestion levels 2 seconds ahead using statistical and machine learning methods, and presents results in a web dashboard powered by Flask. This system serves as a practical demonstration of computer vision and machine learning for traffic monitoring.

![Dashboard Screenshot](https://github.com/Lin172005/Smart-Traffic-Monitoring-and-Prediction-System/blob/main/static/frame.jpg)


## Features
- Detects vehicles (cars, buses, trucks, motorcycles) using YOLOv8 with confidence scores.
- Calculates traffic density based on the full video frame, scaled to vehicles per million pixels.
- Predicts congestion using a moving average (dashboard) and linear regression (separate script).
- Displays results in a user-friendly web interface with annotated video frames.

## Installation

### Prerequisites
- Python 3.8 or higher
- Required libraries: `opencv-python`, `ultralytics`, `torch`, `torchvision`, `numpy`, `pandas`, `flask`, `scikit-learn`

### Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Lin172005/Smart-Traffic-Monitoring-and-Prediction-System.git
   cd Smart-Traffic-Monitoring-and-Prediction-System
