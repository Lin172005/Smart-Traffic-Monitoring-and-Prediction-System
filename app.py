from flask import Flask, render_template
import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
import os

# Explicitly set the static folder to the existing one in the project
app = Flask(__name__, static_folder='static')
model = YOLO("yolov8n.pt")

def get_latest_traffic_data():
    cap = cv2.VideoCapture("traffic_video.mp4")
    if not cap.isOpened():
        print("Error: Could not open video file 'traffic_video.mp4'. Ensure the file exists and path is correct.")
        return {"count": 0, "density": 0, "prediction": 0}

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video. Video might be corrupted or ended.")
        cap.release()
        return {"count": 0, "density": 0, "prediction": 0}

    # Get frame dimensions for full ROI
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi_area = frame_width * frame_height
    print(f"Frame size: {frame_width}x{frame_height}, ROI area: {roi_area}")

    # Detect with confidence threshold
    results = model(frame, conf=0.5)
    vehicle_count = len([box for box in results[0].boxes if box.cls in [2, 7, 5, 0]])  # Include buses and motorcycles
    density = (vehicle_count / roi_area) * 1_000_000 if roi_area > 0 else 0
    print(f"Detected {vehicle_count} vehicles, Density: {density:.2f}")

    # Save annotated frame to the existing static folder
    annotated_frame = results[0].plot()
    static_path = os.path.join(app.static_folder, "frame.jpg")
    print(f"Attempting to save frame to: {static_path}")
    if not os.path.exists(app.static_folder):
        print(f"Warning: Static folder not found at {app.static_folder}. Creating it.")
        os.makedirs(app.static_folder)
    success = cv2.imwrite(static_path, annotated_frame)
    if not success:
        print(f"Error: Failed to save {static_path}. Check write permissions or disk space.")
    else:
        print(f"Frame saved successfully to {static_path}")

    # Load historical data and predict
    df = pd.read_csv("traffic_data.csv") if os.path.exists("traffic_data.csv") else pd.DataFrame(columns=["timestamp", "vehicle_count", "density"])
    if len(df) >= 5:
        prediction = df["density"].tail(5).mean()  # Moving average of last 5 frames
    else:
        prediction = density
    print(f"Prediction: {prediction:.2f}")

    cap.release()
    return {"count": vehicle_count, "density": density, "prediction": prediction}

@app.route("/")
def index():
    data = get_latest_traffic_data()
    return render_template("index.html", data=data)

if __name__ == "__main__":
    app.run(debug=True)