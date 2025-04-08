from flask import Flask, render_template
import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
model = YOLO("yolov8n.pt")

def get_latest_traffic_data():
    cap = cv2.VideoCapture("traffic_video.mp4")
    if not cap.isOpened():
        return {"count": 0, "density": 0, "prediction": 0}

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return {"count": 0, "density": 0, "prediction": 0}

    # Get frame dimensions for full ROI
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi_area = frame_width * frame_height

    # Detect with confidence threshold
    results = model(frame, conf=0.5)
    vehicle_count = len([box for box in results[0].boxes if box.cls in [2, 7, 5, 0]])  # Include buses and motorcycles
    density = (vehicle_count / roi_area) * 1_000_000 if roi_area > 0 else 0

    # Save annotated frame for display with error handling
    annotated_frame = results[0].plot()
    if not os.path.exists("static"):
        os.makedirs("static")
    cv2.imwrite("static/frame.jpg", annotated_frame)

    # Load historical data and predict
    df = pd.read_csv("traffic_data.csv") if os.path.exists("traffic_data.csv") else pd.DataFrame(columns=["timestamp", "vehicle_count", "density"])
    if len(df) >= 5:
        prediction = df["density"].tail(5).mean()  # Moving average of last 5 frames
    else:
        prediction = density

    cap.release()
    return {"count": vehicle_count, "density": density, "prediction": prediction}

@app.route("/")
def index():
    data = get_latest_traffic_data()
    return render_template("index.html", data=data)

if __name__ == "__main__":
    app.run(debug=True)