import cv2
import numpy as np
import sys
import json
from PIL import Image
import os
#base_path = r"G:\Shared drives\BGR Drive\רכב אוטונומי 2024\Autonomous_Divison\PM\Camera\FSOCO_dataset\fsoco_yolov5\Seg_images\test"
label_path = r"G:\Shared drives\BGR Drive\רכב אוטונומי 2024\Autonomous_Divison\PM\Camera\FSOCO_dataset\fsoco_yolov5\Seg_labels\test\ulm_00504.jpg.txt"
#image_path = os.path.join(base_path, "ulm_00504.jpg")
image_path = r"G:\\Shared drives\\BGR Drive\\רכב אוטונומי 2024\\Autonomous_Divison\\PM\\Camera\\FSOCO_dataset\\fsoco_yolov5\\Seg_images\\test\\ulm_00504.jpg"
#try_path = r"C:\Pictures\student_id"

if not os.path.exists(image_path):
    print("File not found!")
# image = Image.open(image_path)
# image.show()
with open(image_path, 'rb') as f:
    print("File is accessible.")
image = cv2.imread(image_path)
with open(label_path, 'r') as file:
    lines = file.readlines()



height, width, _ = image.shape
for line in lines:
    values = list(map(float, line.strip().split()))
    class_id = int(values[0])  # First value is the class ID
    points = values[1:]  # Remaining values are segmentation points

    if len(points) % 2 != 0:
        print(f"Error: Invalid segmentation format in {label_path}")
        continue

    polygon = np.array([(int(x * width), int(y * height)) for x, y in zip(points[0::2], points[1::2])], np.int32)

    # Draw the polygon mask
    cv2.polylines(image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.fillPoly(image, [polygon], color=(0, 255, 0, 50))  # Fill with transparency




