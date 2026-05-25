import cv2
import time
from picamera2 import Picamera2
from ultralytics import YOLO

OUTPUT_PATH = "/home/iot3_user/Pictures/photo1.jpeg"

# Car class
# 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = [2, 3, 5, 7]


model = YOLO("yolov26n.pt")

# setup cams
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 640)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# wait for cam
time.sleep(1)

# Took a photo
frame = picam2.capture_array()
picam2.stop()

# yolo
results = model(frame, imgsz=640, conf=0.4, classes=[2,3,5,7])

object_found = False

for box in results[0].boxes:
    class_id = int(box.cls[0])
    class_name = model.names[class_id]

    if class_id in VEHICLE_CLASSES: #only vehicle
        object_found = True
        print(f"Detected vehicle: {class_name}")

annotated_frame = results[0].plot()

# save image
cv2.imwrite(OUTPUT_PATH, annotated_frame)

print(f"saved:{OUTPUT_PATH}")
print(f"vehicle_found:{object_found}")