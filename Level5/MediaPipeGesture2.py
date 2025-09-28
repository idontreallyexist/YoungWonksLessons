import mediapipe as mp
import cv2
import time
import mediapipe as mp
import numpy as np
import sys
import pandas as pd
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
input=sys.stdin.readline
print=sys.stdout.write
df=pd.DataFrame()
landmarks=["WRIST","THUMB_CMC","THUMB_MCP","THUMB_IP","THUMB_TIP","INDEX_FINGER_MCP"
           "INDEX_FINGER_PIP","INDEX_FINGER_DIP","INDEX_FINGER_TIP","MIDDLE_FINGER_MCP",
           "MIDDLE_FINGER_PIP","MIDDLE_FINGER_DIP","MIDDLE_FINGER_TIP","RING_FINGER_MCP",
           "RING_FINGER_PIP","RING_FINGER_DIP","RING_FINGER_TIP","PINKY_MCP","PINKY_PIP",
           "PINKY_DIP","PINKY_TIP"]
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

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

RESULT = None

# Create a hand landmarker instance with the live stream mode:
def print_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # print(result)
    global RESULT
    RESULT = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result, num_hands=1)

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_np = np.array(frame)
        timestamp = int(round(time.time() * 1000))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_np)
        frame = mp_image.numpy_view()
        landmarker.detect_async(mp_image, timestamp)
        if type(RESULT) is not type(None):
            hand_landmarks_list = RESULT.hand_landmarks
            frame = draw_landmarks_on_image(frame, RESULT)
        else:
            print('else')
        cv2.imshow('Frame', frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('p'):
            landmark_info={}
            for i in enumerate(hand_landmarks_list[0]):
              if i[0]<20:
                landmark_info[landmarks[i[0]]+"x"]=[i[1].x]
                landmark_info[landmarks[i[0]]+"y"]=[i[1].y]
                landmark_info[landmarks[i[0]]+"z"]=[i[1].z]
              else:
                break
            landmark_info["Type"]=0
            df=pd.concat([df,pd.DataFrame(landmark_info)],ignore_index=True)
            print("Paper"+"\n")
        elif key == ord('r'):
            landmark_info={}
            for i in enumerate(hand_landmarks_list[0]):
              if i[0]<20:
                landmark_info[landmarks[i[0]]+"x"]=[i[1].x]
                landmark_info[landmarks[i[0]]+"y"]=[i[1].y]
                landmark_info[landmarks[i[0]]+"z"]=[i[1].z]
              else:
                break
            landmark_info["Type"]=1
            df=pd.concat([df,pd.DataFrame(landmark_info)],ignore_index=True)
            print("Rock"+"\n")
        elif key == ord('s'):
            landmark_info={}
            for i in enumerate(hand_landmarks_list[0]):
              if i[0]<20:
                landmark_info[landmarks[i[0]]+"x"]=[i[1].x]
                landmark_info[landmarks[i[0]]+"y"]=[i[1].y]
                landmark_info[landmarks[i[0]]+"z"]=[i[1].z]
              else:
                break
            landmark_info["Type"]=2
            df=pd.concat([df,pd.DataFrame(landmark_info)],ignore_index=True)
            print("Scissors"+"\n")
    cap.release()
    cv2.destroyAllWindows()
print(str(df))
df.to_csv("rps_data.csv")