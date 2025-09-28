import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import time

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

model_path = 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
RESULT = None

# Create a hand landmarker instance with the live stream mode:
def print_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print(result)
    global RESULT
    RESULT = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

# with HandLandmarker.create_from_options(options) as landmarker:
#     image = mp.Image.create_from_file("Hand_page-0001.jpg")
#     detection_result = landmarker.detect(image)

with HandLandmarker.create_from_options(options) as landmarker:
  cap=cv2.VideoCapture(0)
  while cap.isOpened():
      ret, frame = cap.read()
      if ret:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame_timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, frame_timestamp_ms)
        #if detectResult:
        #  frame = draw_landmarks_on_image(frame, detectResult)
        #cv2.imshow("Video",cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
  cap.release()
  cv2.destroyAllWindows()

#annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
#resize=cv2.resize(annotated_image,dsize=None,fx=0.4,fy=0.4)
#cv2.imshow("MediaPipe",cv2.cvtColor(resize, cv2.COLOR_RGB2BGR))
#cv2.waitKey()
#cv2.destroyAllWindows()