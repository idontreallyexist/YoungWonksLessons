import mediapipe as mp
import cv2
import time
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import random

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)
IMAGE_DIMENSIONS = (640,480)
df=pd.read_csv("rps_data.csv")
landmarks=["WRIST","THUMB_CMC","THUMB_MCP","THUMB_IP","THUMB_TIP","INDEX_FINGER_MCP"
           "INDEX_FINGER_PIP","INDEX_FINGER_DIP","INDEX_FINGER_TIP","MIDDLE_FINGER_MCP",
           "MIDDLE_FINGER_PIP","MIDDLE_FINGER_DIP","MIDDLE_FINGER_TIP","RING_FINGER_MCP",
           "RING_FINGER_PIP","RING_FINGER_DIP","RING_FINGER_TIP","PINKY_MCP","PINKY_PIP",
           "PINKY_DIP","PINKY_TIP"]
columns=[]
for i in landmarks:
    columns.append(i+"x")
    columns.append(i+"y")
    columns.append(i+"z")
x=df[columns]
y=df["Type"]
df2=pd.DataFrame()

model=LogisticRegression(max_iter=500)
xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.2)
model.fit(xTrain,yTrain)

print(xTest)
y_predict=model.predict(xTest)
accuracy=metrics.accuracy_score(yTest,y_predict)
cfMatrix=metrics.confusion_matrix(yTest,y_predict)
print(accuracy)
print(cfMatrix)

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
possible=["Paper","Rock","Scissors"]

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
        elif key == ord('s'):
            landmark_info={}
            for i in enumerate(hand_landmarks_list[0]):
              if i[0]<20:
                landmark_info[landmarks[i[0]]+"x"]=[i[1].x]
                landmark_info[landmarks[i[0]]+"y"]=[i[1].y]
                landmark_info[landmarks[i[0]]+"z"]=[i[1].z]
              else:
                break
            df2=pd.DataFrame(landmark_info)
            y_predict=model.predict(df2)
            comp=random.randint(0,2)
            print("You picked",possible[y_predict[0]])
            print("Computer picked",possible[comp])
            if y_predict[0]==comp:
               print("Tie")
            elif (y_predict[0]==0 and comp==1) or (y_predict[0]==1 and comp==2) or (y_predict[0]==2 and comp==0):
               print("You Win")
            else:
               print("You Lose")
    cap.release()
    cv2.destroyAllWindows()