import cv2
import os
from ultralytics import YOLO

# CONFIG 
IMAGE_DIR = "Dataset/images"
OUTPUT_DIR = "output"
MODEL_PATH = "models/yolov8n.pt"
PERSON_CLASS_ID = 0
CONF_THRESHOLD = 0.25

IMAGE_OUT_DIR = os.path.join(OUTPUT_DIR, "images")
LABEL_OUT_DIR = os.path.join(OUTPUT_DIR, "labels")

os.makedirs(IMAGE_OUT_DIR, exist_ok=True)
os.makedirs(LABEL_OUT_DIR, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Supported image extensions
image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

for image_name in os.listdir(IMAGE_DIR):
    if not image_name.lower().endswith(image_extensions):
        continue

    image_path = os.path.join(IMAGE_DIR, image_name)
    image = cv2.imread(image_path)

    if image is None:
        continue

    h, w, _ = image.shape

    # Run inference
    results = model(image)[0]

    annotation_lines = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id == PERSON_CLASS_ID and conf >= CONF_THRESHOLD:
            x1, y1, x2, y2 = box.xyxy[0]

            # Convert to YOLO format (normalized)
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            box_width = (x2 - x1) / w
            box_height = (y2 - y1) / h

            annotation_lines.append(
                f"{cls_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            )

            # Draw bounding box
            cv2.rectangle(
                image,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )
            cv2.putText(
                image,
                f"Person {conf:.2f}",
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    # Save annotation file
    label_name = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(LABEL_OUT_DIR, label_name)

    with open(label_path, "w") as f:
        f.write("\n".join(annotation_lines))

    # Save annotated image
    output_image_path = os.path.join(IMAGE_OUT_DIR, image_name)
    cv2.imwrite(output_image_path, image)

    print(f"Processed: {image_name}")

print("Batch annotation completed.")
