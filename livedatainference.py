import mediapipe as mp
import cv2
from keras.models import load_model
import logichelper
import numpy as np
import time

from collections import deque
from constants import actions, relevant_hand_landmark_indexes, relevant_pose_landmark_indexes, frames_per_data_set, min_tracking_confidence, min_detection_confidence, cutoff_for_best_prediction
import logichelper

model_average_prediction = logichelper.new_prediction_array()
model_prediction  = logichelper.new_prediction_array()

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

frame_count = 0

cap = cv2.VideoCapture(0)

saved_frames = logichelper.new_saved_frames_queue()
saved_predictions = logichelper.new_saved_predictions_queue()

previous_frame_time = 0


def apply_best_prediction(image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0, 255, 0)
    thickness = 2

    height, width, channels = image.shape

    best_prediction_index = -1
    best_prediction_value = 0

    for prediction_index in range(1, len(model_prediction)):
        prediction = model_average_prediction[prediction_index]
        if prediction >= cutoff_for_best_prediction and prediction > best_prediction_value:
            best_prediction_index = prediction_index
            best_prediction_value = prediction

    if best_prediction_index > -1:
        text = logichelper.decimal_to_percent(best_prediction_value) + ' ' + actions[best_prediction_index]
        cv2.putText(image, text , (12,50), font, fontScale, color, thickness, cv2.LINE_AA)


def apply_prediction_text(image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0, 255, 0)
    thickness = 2
    leftoffset = 20
    rightoffset = 1000

    y_offset = 0

    for i in range(len(actions)):
        cv2.putText(image, actions[i] + ': ' + logichelper.decimal_to_percent(model_average_prediction[i]), (10,40+y_offset), font, fontScale, color, thickness, cv2.LINE_AA)
        y_offset = y_offset + 50

def apply_fps(image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0, 255, 0)
    thickness = 2

    height, width, channels = image.shape

    x_position = width - 100

    time_delta = time.time() - previous_frame_time
    fps = 1 / time_delta
    fps = round(fps,1)
    cv2.putText(image, str(fps), (1200,50), font, fontScale, color, thickness, cv2.LINE_AA)


model = load_model('models/karate-kid.h5')

with mp_holistic.Holistic(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence) as holistic:

    previous_frame_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()

        #recolor feed
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        #make detections
        results = holistic.process(image)

        #recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #draw right hand features
        lhl = results.left_hand_landmarks
        mp_drawing.draw_landmarks(image, lhl, mp_holistic.HAND_CONNECTIONS)

        rhl = results.right_hand_landmarks
        mp_drawing.draw_landmarks(image, rhl, mp_holistic.HAND_CONNECTIONS)

        #draw pose features
        pl = results.pose_landmarks
        mp_drawing.draw_landmarks(image, pl, mp_holistic.POSE_CONNECTIONS)


        if not logichelper.should_skip_frame(frame_count):

            landmarks_by_feature = [[lhl,relevant_hand_landmark_indexes],[rhl,relevant_hand_landmark_indexes],[pl,relevant_pose_landmark_indexes]]
            landmarks_for_frame = []

            for each in landmarks_by_feature:
                landmarks_for_frame = landmarks_for_frame + logichelper.collect_relevant_landmarks(each[0], each[1])

            saved_frames.popleft()
            saved_frames.append(landmarks_for_frame)

            model_prediction = model.predict([logichelper.flatten_queue(saved_frames)], batch_size=1, verbose=0)[0].tolist()

            saved_predictions.popleft()
            saved_predictions.append(model_prediction)

            model_average_prediction = logichelper.calculate_average_predictions(saved_predictions)



        #apply_prediction_text(image)
        #apply_fps(image)
        apply_best_prediction(image)
        previous_frame_time = time.time()

        cv2.imshow('Live Action Inference' , image)

        frame_count = frame_count + 1

        keyPress = cv2.waitKey(1)

        if keyPress == ord('q'):
            break



cap.release()
cv2.destroyAllWindows()
