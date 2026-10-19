from ultralytics import YOLO
import cv2
import time

# YOLO model
model = YOLO("yolo11n.pt")

# Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" Camera open nahi ho raha.")
    exit()
    
    cv2.namedWindow("live object detector",cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

# FPS calculation
prev_time = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera frame nahi mil raha.")
        break

    # Object detection
    results = model(frame, conf=0.40, verbose=False)

    # Draw detections
    output = results[0].plot()

    # Current FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # Number of detected objects
    boxes = results[0].boxes
    object_count = len(boxes)

    # FPS and count
    cv2.putText(
        output,
        f"FPS: {fps:.1f}",
        (40,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        output,
        f"Objects: {object_count}",
        (40, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 250, 0),
        2
    )

    # Show detected objects
    cv2.imshow("Live Object Detector", output)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera
cap.release()
cv2.destroyAllWindows() 