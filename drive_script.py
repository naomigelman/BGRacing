import os
import shutil
from math import ceil

# Path to the main directory
main_dir = r"C:\Users\97254\Pictures\Formula\fsoco_segmentation_train_extracted"


#Checking that all images in the "img" folders has a matching file in the "ann" folder with the same name.
#removing extention (after".") to comapre only file name.
#making sure there are no extra files in each subfolder
missmatch = False
for team in os.listdir(main_dir):
    team_path = os.path.join(main_dir, team)
    if os.path.isdir(team_path):
        ann_path = os.path.join(team_path , "ann")
        img_path = os.path.join(team_path , "img")
        ann_files = {f.split(".")[0] for f in os.listdir(ann_path)}
        img_files = {f.split(".")[0] for f in os.listdir(img_path)}

        for ann_file in ann_files:
            # print(f"name is {ann_file}")
            if not ann_file in img_files:
                missmatch = True
                print(f"file: {ann_file} is missing in img for team {team}")


#if all files are matching- split every subfolder into 1/3 and divide them between test, val, train

if not missmatch:
    drive_path = r"G:\Shared drives\BGR Drive\רכב אוטונומי 2024\Autonomous_Divison\PM\Camera\FSOCO_dataset\fsoco_yolov5"
    drive_images = os.path.join(drive_path, "Seg_images")
    drive_labels = os.path.join(drive_path, "Seg_labels")

    # Define the subfolders for splits
    splits = ["train", "val", "test"]

    for team in os.listdir(main_dir):
        team_path = os.path.join(main_dir, team)
        if os.path.isdir(team_path):
            ann_path = os.path.join(team_path, "ann")
            img_path = os.path.join(team_path, "img")

            img_files = sorted(os.listdir(img_path))
            ann_files = sorted(os.listdir(ann_path))

            total_files = len(img_files)
            split_sizes = [ceil(total_files / 3)] * 3

            # Create index ranges for train, val, test
            train_idx = slice(0, split_sizes[0])
            val_idx = slice(split_sizes[0], split_sizes[0] + split_sizes[1])
            test_idx = slice(split_sizes[0] + split_sizes[1], total_files)

            indices = [train_idx, val_idx, test_idx]

            # Copy files into the respective folders
            for split, idx in zip(splits, indices):
                split_img_dir = os.path.join(drive_images, split)
                split_ann_dir = os.path.join(drive_labels, split)

                os.makedirs(split_img_dir, exist_ok=True)
                os.makedirs(split_ann_dir, exist_ok=True)

                for i in range(*idx.indices(total_files)):  # Convert slice to range
                    img_src = os.path.join(img_path, img_files[i])
                    ann_src = os.path.join(ann_path, ann_files[i])

                    img_dst = os.path.join(split_img_dir, img_files[i])
                    ann_dst = os.path.join(split_ann_dir, ann_files[i])

                    shutil.copy(img_src, img_dst)
                    shutil.copy(ann_src, ann_dst)
