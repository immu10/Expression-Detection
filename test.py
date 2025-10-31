# import cv2


# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     # fps = cap.get(cv2.CAP_PROP_FPS)
#     # print("FPS:", fps)
#     if ret:


        
#         cv2.imshow("Frame", frame)
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break



# import cv2
# import mediapipe as mp

# # Initialize MediaPipe Face Detection (BlazeFace)
# mp_face_detection = mp.solutions.face_detection
# mp_drawing = mp.solutions.drawing_utils

# # Open webcam stream
# cap = cv2.VideoCapture(0)

# # Initialize the face detection model
# with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame.")
#             break

#         # Convert the frame to RGB (MediaPipe expects RGB)
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Process the frame with MediaPipe
#         results = face_detection.process(rgb_frame)

#         # Draw detections on the frame
#         if results.detections:
#             for detection in results.detections:
#                 mp_drawing.draw_detection(frame, detection)

#         # Show the frame
#         cv2.imshow('BlazeFace Face Tracking', frame)

#         # Press 'q' to quit
#         if cv2.waitKey(5) & 0xFF == ord('q'):
#             break

# # Clean up
# cap.release()
# cv2.destroyAllWindows()


import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = r"I:\codebs\MediapipeModel\blaze_face_short_range.tflite"
# --- Load the MediaPipe BlazeFace model ---
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,  # Enables async detection
    result_callback=lambda result, output_image, timestamp_ms: on_result(result, output_image, timestamp_ms)
)

# Callback for async results
def on_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

# Initialize detector
detector = vision.FaceDetector.create_from_options(options)

# Open webcam
cap = cv2.VideoCapture(0)

latest_result = None
timestamp = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Send frame to async detector
    detector.detect_async(mp_image, timestamp)
    timestamp += 33  # ~30 FPS (in ms)

    # Draw previous detection results (if any)
    if latest_result and latest_result.detections:
        for detection in latest_result.detections:
            bbox = detection.bounding_box
            cv2.rectangle(frame,
                          (bbox.origin_x, bbox.origin_y),
                          (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                          (0, 255, 0), 2)

    cv2.imshow("MediaPipe BlazeFace (Async)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
