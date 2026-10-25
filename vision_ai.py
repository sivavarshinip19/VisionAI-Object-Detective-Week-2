
from ultralytics import YOLO
import cv2

print("Loading VisionAI Object Detective...")

model = YOLO("yolo11s.pt")

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("VisionAI started! Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read webcam frame.")
        break

    results = model.predict(
    source=frame,
    conf=0.15,
    imgsz=960,
    verbose=False
)

    output = results[0].plot()

    cv2.imshow("VisionAI - Object Detective | Week 2", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
