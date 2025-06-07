from ultralytics import YOLO
import torch
import cv2
import os
import yaml
import numpy as np
from IPython.display import Video, display

class Detect_And_Track:
    def __init__(self, model, device, bytetrack_path=r"C:\Users\97254\PycharmProjects\pythonProject1\BGR\detection_v8\bytetrack.yaml"):
        self.model = model
        self.device = device
        self.bytetrack_path = bytetrack_path
        self.class_colors = {
            0: (255, 0, 0),
            1: (0, 200, 200),
            2: (255, 165, 0),
            3: (128, 0, 128),
            4: (0, 255, 255)
        }

    def track(self, frame, conf=0.1, iou=0.25):
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        results = self.model.track(
            source=frame_tensor,
            persist=True,
            conf=conf,
            iou=iou,
            tracker=self.bytetrack_path,
            stream=True
        )
        return next(results).boxes

    def process_video(self, output_txt, input_path):
        cap = cv2.VideoCapture(input_path)

        W, H = 640, 480
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_limit = int(fps * 10)  # 10 seconds - Only for testing! remove for full video

        frame_count = 0

        print(f"Writing detections to {output_txt}...")

        with open(output_txt, "w") as file:
            #headers of each column
            file.write("frame,class_id,track_id,num_cones,x1,y1,x2,y2\n")

            while cap.isOpened() and frame_count < frame_limit:
                ret, frame = cap.read()
                if not ret:
                    break

                boxes = self.track(frame)
                num_cones = len(boxes)
                frame_count += 1

                for box in boxes:
                    class_id = int(box.cls[0])
                    track_id = int(box.id[0]) if box.id is not None else -1
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    file.write(f"{frame_count},{class_id},{track_id},{num_cones},{x1},{y1},{x2},{y2}\n")

        cap.release()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = r"C:\Users\97254\PycharmProjects\pythonProject1\BGR\detection_v8\nano_best.pt"
    bytetrack_path = r"C:\Users\97254\PycharmProjects\pythonProject1\BGR\detection_v8\byetrack.yaml"
    output_path = r"C:\Users\97254\PycharmProjects\pythonProject1\BGR\detection_v8\results.txt"

    model = YOLO(model_path).to(device)

    DAT = Detect_And_Track(model, device, bytetrack_path)

    input_video = r"C:\Users\97254\PycharmProjects\pythonProject1\BGR\BGR-PM\Sensors\Camera\ut\record_example\record_motorcity.avi"


    DAT.process_video(output_path, input_video)

if __name__ == "__main__":
    main()
