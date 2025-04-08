import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd

# Load YOLO model
model = YOLO("yolov8n.pt")  # Nano model for speed

# Load video
cap = cv2.VideoCapture("traffic_video.mp4")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
roi_area = frame_width * frame_height

# Store data
data = {"timestamp": [], "vehicle_count": [], "density": []}
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Detect with confidence threshold
    results = model(frame, conf=0.5)
    vehicle_count = len([box for box in results[0].boxes if box.cls in [2, 7, 5, 0]])  # 2=car, 7=truck, 5=bus, 0=motorcycle

    # Use entire frame as ROI
    density = (vehicle_count / roi_area) * 1_000_000 if roi_area > 0 else 0

    # Log data
    data["timestamp"].append(frame_count / 30)  # Assuming 30 FPS
    data["vehicle_count"].append(vehicle_count)
    data["density"].append(density)

    # Display
    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, f"Vehicles: {vehicle_count} | Density: {density:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Traffic Monitor", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("traffic_data.csv", index=False)
print("Data saved to traffic_data.csv")