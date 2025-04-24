
# ClassId in orig labels:
# Blue cone: 2926562
# Yellow cone: 2926555
# Orange cone: 2926554
# large_orange_cone: 2926561
# unknown_cone: 2926563
#
# in new labels:
# Blue cone: 0
# Yellow cone: 1
# Orange cone: 2
# large_orange_cone: 3
# unknown_cone: 4

import os

# Your mapping
id_map = {
    "2926562": "0",  # Blue cone
    "2926555": "1",  # Yellow cone
    "2926554": "2",  # Orange cone
    "2926561": "3",  # Large orange cone
    "2926563": "4",  # Unknown cone
}


labels_dir = r"G:\Shared drives\BGR Drive\רכב אוטונומי 2024\Autonomous_Divison\PM\Camera\FSOCO_dataset\fsoco_yolov5\Segmentation\labels\train"
#specific_path = r"G:\Shared drives\BGR Drive\רכב אוטונומי 2024\Autonomous_Divison\PM\Camera\FSOCO_dataset\fsoco_yolov5\Segmentation\labels\test\amz_00630.jpg.txt"

for fname in os.listdir(labels_dir):
    if not fname.endswith(".txt"):
        continue
    path = os.path.join(labels_dir, fname)
    new_lines = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] not in id_map:
                continue  # Skip unknown classes
            new_class_id = id_map[parts[0]]
            new_line = " ".join([new_class_id] + parts[1:])
            new_lines.append(new_line)
    with open(path, "w") as f:
        f.write("\n".join(new_lines) + "\n")
