#script that removes .png and .jpg to the annotation label files.

import os
import glob

# Folder where your label files are
labels_dir = r"G:\Shared drives\BGR Drive\רכב אוטונומי 2024\Autonomous_Divison\PM\Camera\FSOCO_dataset\fsoco_yolov5\Segmentation\labels\val"

# Find all .png.txt and .jpg.txt files
pattern_png = os.path.join(labels_dir, "*.png.txt")
pattern_jpg = os.path.join(labels_dir, "*.jpg.txt")
files = glob.glob(pattern_png) + glob.glob(pattern_jpg)

print(f"Found {len(files)} label files to rename.")

# Rename each file
for old_path in files:
    new_path = old_path.replace(".png.txt", ".txt").replace(".jpg.txt", ".txt")
    os.rename(old_path, new_path)
    print(f"Renamed: {old_path} -> {new_path}")

print("✅ Done renaming all label files!")
