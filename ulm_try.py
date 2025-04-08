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
ANNOTATIONS_PATH = r"C:\Users\97254\Pictures\Formula\ulm_00504.jpg.json"
OUTPUT_LABELS_PATH = r"C:\Users\97254\Pictures\Formula\txt_result"
IMAGE_PATH = r"C:\Users\97254\Pictures\Formula\ulm_00504.jpg"
image = cv2.imread(r"C:\Users\97254\Pictures\Formula\ulm_00504.jpg")
IMAGE_HEIGHT, IMAGE_WIDTH, channels = image.shape

# Ensure output directory exists
os.makedirs(OUTPUT_LABELS_PATH, exist_ok=True)


def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.fromstring(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return mask

def extract_polygons(mask):
    """Extract contours (polygon points) from a binary mask."""
    mask = np.uint8(mask) * 255  # Convert boolean mask to uint8 (True -> 255, False -> 0)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def convert_to_yolo_format(contours, image_width, image_height, origin):
    """Convert OpenCV contours to YOLO segmentation format."""
    yolo_annotations = []
    for contour in contours:
        # Normalize each (x, y) point to be between 0 and 1
        points = []
        for point in contour:
            x, y = point[0]
            x = (x + origin[0]) / image_width
            y = (y + origin[1]) / image_height
            points.append(f"{x:.6f} {y:.6f}")

        if len(points) >= 6:  # YOLO segmentation requires at least 3 points (x, y pairs)
            yolo_annotations.append(" ".join(points))

    return yolo_annotations

def process_annotation(json_path):
    """Convert a Supervisely JSON annotation file to YOLO format."""
    with open(json_path, "r") as file:
        data = json.load(file)
        #print(json.dumps(data, indent=4))  # Pretty-print JSON
    image_filename = os.path.splitext(os.path.basename(json_path))[0]
    output_txt_path = os.path.join(OUTPUT_LABELS_PATH, f"{image_filename}.txt")

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


process_annotation(ANNOTATIONS_PATH)

print("Conversion completed! YOLO labels saved in:", OUTPUT_LABELS_PATH)


