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
cap = cv2.VideoCapture(0)
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult

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


def result_callback(result, output_image, timestamp_ms):
    # Handle each frame's result here
    print("Detected faces:", result.detections)


def visualize(image,detection_result):
    height, width, _ = image.shape
    # annotated_image = image.copy()
    # pprint.pprint(detection_result.detections)
    # print(detection_result.detections[0].keypoints,"1\n")
    for detection in detection_result.detections:
        # print(detection,"2\n")
        # Draw bounding box
        box = detection.bounding_box
        start_point = (int(box.origin_x), int(box.origin_y))
        end_point = (int(box.origin_x + box.width), int(box.origin_y + box.height))
        cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
        
        # Draw keypoints
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                     width, height)
            color, thickness, radius = (0, 255, 0), 2, 2
            cv2.circle(image, keypoint_px, thickness, color, radius)
    
    return image

# image = mp.Image.create_from_file(image_path)

def print_result(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
    print('face detector result: {}'.format(result))

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(base_options=base_options,running_mode=vision.RunningMode.LIVE_STREAM, result_callback=callback)
detector = vision.FaceDetector.create_from_options(options)
# frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_rate = 30
frame_idx = 0
cv2.namedWindow("name", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    timestamp = frame_idx * (1000 / frame_rate)

    
    if ret:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        detection_result = detector.detect_async(mp_image,int(timestamp))
        if detection_result is None:
            
            continue

        # image_copy = np.copy(frame.numpy_view())
        annotated_image = visualize(frame, detection_result)
        # rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        cv2.imshow("name",annotated_image)
        
        
        # cv2.imshow("Frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        # detection_result = detector.detect(image)
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


