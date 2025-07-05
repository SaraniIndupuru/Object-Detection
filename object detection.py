import cv2
#import numpy as np
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO('yolov8n.pt')
class_names = model.names

# Initialize Deep SORT tracker
tracker = DeepSort(max_age=30)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize video writer (optional)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Start frame processing
prev = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 model
    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = class_names[cls]
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracking results
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = track.to_ltwh()
        r, b = int(l + w), int(t + h)

        cv2.rectangle(frame, (int(l), int(t)), (r, b), (255, 0, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show FPS
    now = time.time()
    fps = 1 / (now - prev)
    prev = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display output
    cv2.imshow("Object Detection and Tracking", frame)
    out.write(frame)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()