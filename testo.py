import mediapipe as mp
import numpy as np
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pprint
import math
from typing import Tuple, Union

model_path = r"I:\codebs\MediapipeModel\blaze_face_short_range.tflite"
# image_path = r"WIN_20251025_16_51_38_Pro.jpg"
image_path = r"Screenshot 2025-10-27 000158.png"
img = cv2.imread(image_path)
cap = cv2.VideoCapture(0)
# FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
new_frame = None

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


# def result_callback(result, output_image, timestamp_ms):
#     # Handle each frame's result here
#     print("Detected faces:", result.detections)


def visualize(result,output_image: mp.Image,timestamp_ms: int):
    global detection_result,new_frame
    new_frame = output_image
    # height, width, _ = image.shape
    
    detection_result = result
    # print(result)

    # annotated_image = image.copy()
    # pprint.pprint(detection_result.detections)
    # print(detection_result.detections[0].keypoints,"1\n")
   
        
        
    
    # return image
    # print_result(result)
# image = mp.Image.create_from_file(image_path)

# def print_result(result, output_image: mp.Image, timestamp_ms: int):
#     # output_image = visualize(output_image.numpy_view(),result) 
#     # # new_frame = visualize(output_image.numpy_view(),result)
#     # annotated_image = visualize(output_image, result)
#     # # rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
#     # cv2.imshow("name",frame.numpy_view())
#     print('face detector result: {}'.format(result))
#     global new_frame

#     new_frame = visualize(detection_result=result)
#     # cv2.imshow("name", visualized_image)
#     # if cv2.waitKey(1) & 0xFF == ord('q'):
#     #         exit(0)
    

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
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # cv2.imshow("name", mp_image.numpy_view())
        # print("checkpoint")
        detector.detect_async(mp_image,int(timestamp))
        # print("checkpoint 2")

        # # image_copy = np.copy(frame.numpy_view())
        # # annotated_image = visualize(frame, detection_result)
        # # rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        # # cv2.imshow("name",annotated_image)
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

                    # keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                    #                                         width, height)
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




# import mediapipe as mp

# BaseOptions = mp.tasks.BaseOptions
# FaceDetector = mp.tasks.vision.FaceDetector
# FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
# VisionRunningMode = mp.tasks.vision.RunningMode

# # Create a face detector instance with the video mode:
# options = FaceDetectorOptions(
#     base_options=BaseOptions(model_asset_path='/path/to/model.task'),
#     running_mode=VisionRunningMode.VIDEO)
# with FaceDetector.create_from_options(options) as detector:
#   # The detector is initialized. Use it here.
#   # ...
#     pass


