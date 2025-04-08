import cv2
import csv
from datetime import datetime
import os

def get_next_index(base_path, base_name, extension):
    index = 0
    while True:
        file_path = f"{base_name}_{index}.{extension}"
        if not os.path.exists(os.path.join(base_path, file_path)):
            break
        index += 1
    return index

# Define the base path and base name for the CSV file
base_path = '.'  # Current directory
base_name = 'frame_data'
extension = 'csv'

index = get_next_index(base_path, base_name, extension)
csv_file_path = os.path.join(base_path, f"{base_name}_{index}.{extension}")
run_folder = os.path.join(base_path, f"{base_name}_{index}") #create the new folder with the running index
os.makedirs(run_folder, exist_ok=True)

vidcap = cv2.VideoCapture(0)  # Change camera config here
success, image = vidcap.read()
count = 0

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame Number', 'Timestamp'])
    while success:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        frame_path = os.path.join(run_folder, f"frame{count}.jpg")
        cv2.imwrite(frame_path, image)
        writer.writerow([count, timestamp])
        success, image = vidcap.read()
        print(f'Read a new frame: {success}, Frame Number: {count}, Timestamp: {timestamp}')

        count += 1
