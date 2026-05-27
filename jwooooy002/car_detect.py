from picamera2 import Picamera2
from time import sleep
from pathlib import Path
from ultralytics import YOLO
import cv2

BASE_DIR = Path("/home/iot3_user/Desktop/iot3/IoT26-HW06/jwooooy002")
IMAGE_DIR = BASE_DIR / "images"

INPUT_IMAGE = IMAGE_DIR / "input.jpg"
RESULT_IMAGE = IMAGE_DIR / "result.jpg"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# 1. 카메라로 사진 촬영
picam2 = Picamera2()
config = picam2.create_still_configuration()
picam2.configure(config)

picam2.start()
sleep(2)

picam2.capture_file(str(INPUT_IMAGE))
picam2.stop()

# 2. YOLO 모델 로드
model = YOLO("yolo26n.pt")

# 3. 사진에서 객체 인식
results = model.predict(
    source=str(INPUT_IMAGE),
    conf=0.25,
    imgsz=640
)

# 4. 자동차 관련 클래스만 확인
vehicle_classes = {"car", "bus", "truck", "motorcycle"}

detected = []

for result in results:
    boxes = result.boxes

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]

            if class_name in vehicle_classes:
                detected.append((class_name, conf))

    # 5. 박스 그려진 결과 이미지 저장
    result.save(filename=str(RESULT_IMAGE))

# 6. Node-RED exec 노드가 받을 출력
if detected:
    print("Vehicle detected:")
    for name, conf in detected:
        print(f"{name}: {conf:.2f}")
else:
    print("No vehicle detected")
