#!/usr/bin/env python3
"""
CODE 1 - Detection only.
YOLOv11m detects people (person class only) and draws boxes. No tracking,
no entry counting, no interaction. Just detection on every frame.
"""

import cv2
import torch
from ultralytics import YOLO
import supervision as sv

# ----------------------------- CONFIG ---------------------------------
SOURCE_VIDEO_PATH = "CRK01.mp4"
TARGET_VIDEO_PATH = "detection_only.mp4"
MODEL_NAME        = "best 26m.pt"
PERSON_CLASS_ID   = 0
CONF              = 0.1      # detection confidence threshold
IMG_SIZE          = 640
DEVICE            = "cuda:0" if torch.cuda.is_available() else "cpu"
HALF              = torch.cuda.is_available()
# ----------------------------------------------------------------------


def main():
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    else:
        print("[GPU] running on CPU.")

    model = YOLO(MODEL_NAME)
    model.to(DEVICE)
    model.fuse()

    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    generator  = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    frame_idx = 0
    with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
        for frame in generator:
            results = model(frame, conf=CONF, classes=[PERSON_CLASS_ID],
                            imgsz=IMG_SIZE, device=DEVICE, half=HALF, verbose=False)
            boxes = results[0].boxes
            for b, c in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(frame, f"person {c:.2f}", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
            sink.write_frame(frame)
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames")

    print(f"Done. Output: {TARGET_VIDEO_PATH}")


if __name__ == "__main__":
    main()