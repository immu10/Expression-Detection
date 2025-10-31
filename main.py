import mediapipe as mp
import numpy as np
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pprint
import math
from typing import Tuple, Union

model_path = r"I:\codebs\MediapipeModel\blaze_face_short_range.tflite"
image_path = r"Screenshot 2025-10-27 000158.png"
img = cv2.imread(image_path)
cap = cv2.VideoCapture(0)
new_frame = None

def visualize(result,output_image: mp.Image,timestamp_ms: int):
    global detection_result,new_frame
    new_frame = output_image
    # height, width, _ = image.shape
    
    detection_result = result
    # print(result)
    

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(base_options=base_options,running_mode=vision.RunningMode.LIVE_STREAM, result_callback=visualize)
detector = vision.FaceDetector.create_from_options(options)
# frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_rate = 30
frame_idx = 0
cv2.namedWindow("name", cv2.WINDOW_NORMAL)
cv2.namedWindow("input", cv2.WINDOW_NORMAL)

while True:
    
    ret, frame = cap.read()
   
    
    
    
    # print("checkpoint 1")
    timestamp = frame_idx * (1000 / frame_rate)
    if ret:
        cv2.imshow("input",frame)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # didn't need
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # print("checkpoint")
        detector.detect_async(mp_image,int(timestamp))
        # print("checkpoint 2")

        if new_frame is not None and detection_result is not None and detection_result.detections:
            for detection in detection_result.detections:
                # print(detection,"2\n")
                global start_point,end_point
                # Draw bounding box
                box = detection.bounding_box
                # print(box,"box\n")
                x,y,w,h = box.origin_x,box.origin_y,box.width,box.height
                
                start_point = (int(x), int(y))
                end_point = (int(x+w), int(y+h))
                # print(start_point,end_point,"points\n")
                #  Draw keypoints
                for keypoint in detection.keypoints:
                    width, height, _ = frame.shape
                    width, height = int(width), int(height)
                    cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
                    color, thickness, radius = (0, 255, 0), 2, 7
                    print(keypoint,"keypoint\n")
                    x = int(keypoint.x * height)
                    y = int(keypoint.y * width)
                    cv2.circle(frame, (x,y), thickness, color, radius)
            cv2.imshow("name", frame)
            new_frame = None
        
        # # cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # # detection_result = detector.detect(image)
        frame_idx += 1





