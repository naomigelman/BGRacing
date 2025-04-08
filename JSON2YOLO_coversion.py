import io
import os
import json
import zlib

import cv2
import numpy as np
import base64
from pathlib import Path
from PIL import Image

# Paths
ANNOTATIONS_DIR = r"G:\Shared drives\BGR Drive\רכב אוטונומי 2024\Autonomous_Divison\PM\Camera\FSOCO_dataset\fsoco_yolov5\Seg_labels\test_JSON"
OUTPUT_LABELS_DIR = r"G:\Shared drives\BGR Drive\רכב אוטונומי 2024\Autonomous_Divison\PM\Camera\FSOCO_dataset\fsoco_yolov5\Seg_labels\test"
IMAGE_DIR = r"G:\Shared drives\BGR Drive\רכב אוטונומי 2024\Autonomous_Divison\PM\Camera\FSOCO_dataset\fsoco_yolov5\Seg_images\test"


# Ensure output directory exists
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)


def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.fromstring(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return mask


def mask_2_base64(mask):
    img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
    img_pil.putpalette([0,0,0,255,255,255])
    bytes_io = io.BytesIO()
    img_pil.save(bytes_io, format='PNG', transparency=0, optimize=0)
    bytes = bytes_io.getvalue()
    return base64.b64encode(zlib.compress(bytes)).decode('utf-8')


def extract_polygons(mask):
    mask = np.uint8(mask) * 255  # Convert boolean mask to uint8 (True -> 255, False -> 0)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def convert_to_yolo_format(contours, image_width, image_height, origin):
    yolo_annotations = []
    for contour in contours:
        # Normalize each (x, y) point to be between 0 and 1
        points = []
        for point in contour:
            x, y = point[0]
            x = (x + origin[0]) / image_width
            y = (y + origin[1]) / image_height
            points.append(f"{x:.6f} {y:.6f}")

        if len(points) >= 6:
            yolo_annotations.append(" ".join(points))

    return yolo_annotations

def process_annotation(json_path, image_path):
    image = cv2.imread(image_path)
    IMAGE_HEIGHT, IMAGE_WIDTH, _ = image.shape
    with open(json_path, "r") as file:
        data = json.load(file)
        #print(json.dumps(data, indent=4))  # Pretty-print JSON
    image_filename = os.path.splitext(os.path.basename(json_path))[0]
    output_txt_path = os.path.join(OUTPUT_LABELS_DIR, f"{image_filename}.txt")

    with open(output_txt_path, "w") as txt_file:
        for obj in data["objects"]:
            if obj["geometryType"] == "bitmap":
                mask = base64_2_mask(obj["bitmap"]["data"])
                origin = obj["bitmap"]["origin"]
                contours = extract_polygons(mask)
                yolo_annotations = convert_to_yolo_format(contours, IMAGE_WIDTH, IMAGE_HEIGHT, origin)

                for annotation in yolo_annotations:
                    class_id = obj["classId"]  # Use classId or map to YOLO class index
                    txt_file.write(f"{class_id} {annotation}\n")





for filename in os.listdir(ANNOTATIONS_DIR):
    if filename.endswith(".png.json"):
        image_path = os.path.join(IMAGE_DIR, filename.removesuffix(".json"))
        process_annotation(os.path.join(ANNOTATIONS_DIR, filename), image_path)

print("Conversion completed! YOLO labels saved in:", OUTPUT_LABELS_DIR)
