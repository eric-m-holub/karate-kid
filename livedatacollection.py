import mediapipe as mp
import numpy as np
import cv2
import json
import logichelper
import time

from constants import actions, relevant_hand_landmark_indexes, relevant_pose_landmark_indexes, min_tracking_confidence, min_detection_confidence



active_training_label = -1
frame_count = 0

                    #SAND THE FLOOR                                 #WAX ON / WAX OFF
#[ [[x1, y1, z1, x2, ...], [x1, y1, z1, x2, ...]], [[x1, y1, z1, x2, ...], [x1, y1, z1, x2, ...]], ... ]
frames_by_label = logichelper.new_frame_data_array()


def apply_active_training_label(image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (0, 255, 0)
    thickness = 2
    if active_training_label > -1:
        label = actions[active_training_label]
    else:
        label = ''
    image = cv2.putText(image, label, org, font, fontScale, color, thickness, cv2.LINE_AA)

def apply_body_box(image, landmarks):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0, 0, 255)

    thickness = 2
    boxoffset = 200
    height, width, channels = image.shape

    x = [landmark.x for landmark in landmarks]
    y = [landmark.y for landmark in landmarks]

    center = np.array([np.mean(x)*width, np.mean(y)*height]).astype('int32')
    if active_training_label > -1:
        text = 'TRAINING ' + actions[active_training_label]
    else:
        text = 'TRAINING'

    cv2.putText(image, text, (center[0]-boxoffset,center[1]-boxoffset-25), font, fontScale, color, 2, cv2.LINE_AA)
    cv2.rectangle(image, (center[0]-boxoffset,center[1]-boxoffset), (center[0]+boxoffset,center[1]+boxoffset), color, thickness)





mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence) as holistic:

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


        if not logichelper.should_skip_frame(frame_count) and active_training_label > -1:
            landmarks_by_feature = [[lhl,relevant_hand_landmark_indexes],[rhl,relevant_hand_landmark_indexes],[pl,relevant_pose_landmark_indexes]]
            landmarks_for_frame = []

            for each in landmarks_by_feature:
                landmarks_for_frame = landmarks_for_frame + logichelper.collect_relevant_landmarks(each[0], each[1])

            frames_by_label[active_training_label][-1].append(landmarks_for_frame)


        apply_active_training_label(image)
        cv2.imshow('Action Recognition Training' , image)

        frame_count = frame_count + 1

        keyPress = cv2.waitKey(1)

        if keyPress == ord('q'):
            break
        elif keyPress == ord('0'):
            active_training_label = -1

        for i in range(len(actions)):
            if keyPress == ord(str(i+1)) and active_training_label != i:
                frames_by_label[i].append([])
                active_training_label = i
                break



cap.release()
cv2.destroyAllWindows()


with open('data/karate-kid.json') as json_file:
    json_data = json.load(json_file)
    if len(json_data) < 1:
        json_data = frames_by_label
    else:
        for i in range(len(json_data)):
            json_data[i] = json_data[i] + frames_by_label[i]

    with open('data/karate-kid.json', 'w') as outfile:
        json.dump(json_data, outfile)
        #import modelgenerator
