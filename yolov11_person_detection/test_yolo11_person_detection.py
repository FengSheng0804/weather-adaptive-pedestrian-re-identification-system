# 使用YOLOv11模型进行行人检测，并保存处理后的视频

import cv2
from ultralytics import YOLO

model = YOLO("./weights/yolo11s.pt")
cap = cv2.VideoCapture('people.mp4')

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

PAD = 50  # pixels added on each side

def pad_frame(frame, pad):
    return cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))

def unpad_frame(frame, pad):
    return frame[pad:-pad, pad:-pad]
    
output_path = "output_video.mp4"  # Path to save the processed video
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
   ret, frame = cap.read()
   if not ret:
       break
   padded_frame = pad_frame(frame,PAD)
   results = model.track(padded_frame, task="Detection", classes=[0])
   plotted = results[0].plot(labels=False, conf=False)
   unpaded_frame = unpad_frame(plotted,PAD)
   out.write(unpaded_frame)

cap.release()
out.release()