from math import sqrt
from collections import deque
import numpy as np

from constants import perform_z_normalization, z_normalization_absolute_max, frame_skip, actions, relevant_hand_landmark_indexes, relevant_pose_landmark_indexes, frames_per_data_set

def new_frame_data_array():
    ret = []
    for i in range(len(actions)):
        ret.append([])
    return ret

def new_saved_frames_queue():
    dimensions = data_input_dimensions()
    dimensions = int(dimensions / frames_per_data_set)
    ret = deque()

    for i in range(frames_per_data_set):
        frame = []
        for i in range(dimensions):
            frame = frame + [0.0]
        ret.append(frame)
    return ret

def new_saved_predictions_queue():
    ret = deque()

    for i in range(frames_per_data_set):
        prediction = []
        for j in range(len(actions)):
            prediction = prediction + [0.0]
        ret.append(prediction)
    return ret

def new_prediction_array():
    ret = []
    for _ in range(len(actions)):
        ret.append(0.0)
    return ret

def calculate_average_predictions(saved_predictions):
    number_of_actions = len(actions)
    ret = new_prediction_array()

    for saved_prediction in saved_predictions:
        ret = np.add(ret, saved_prediction)

    return np.divide(ret, [len(saved_predictions)])


def flatten_queue(queue):
    ret = []
    for each in queue:
        ret = ret + each
    return ret

def decimal_to_percent(decimal):
    percent = decimal*100
    percent = round(percent)
    return str(percent) + '%'

def distance_between_vectors(p1, p2):
    return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)

def data_input_dimensions():
    return (len(relevant_hand_landmark_indexes)*2+len(relevant_pose_landmark_indexes))*3*frames_per_data_set


def should_skip_frame(frame_count):
    return frame_count % frame_skip != 0


def landmark_to_vector(landmark):
    return [landmark.x, landmark.y, landmark.z]

def zero_vector():
    return [0.0, 0.0, 0.0]

def array_zero_vectors(n):
    ret = []
    for i in range(n):
        ret = ret + zero_vector()
    return ret


def normalize_z_value(z):
    ret = 0

    ret = z + z_normalization_absolute_max
    ret = ret / (z_normalization_absolute_max*2)

    if ret < 0.0:
        return 0.0
    elif ret > 1.0:
        return 1.0

    return ret

# [x1, y1, z1, x2, y2, z2, x3, y3, z3, ...]
def relevant_landmarks(landmarks_to_target, landmarks):
    ret = []

    for landmark_index in landmarks_to_target:
        landmarkvector = landmark_to_vector(landmarks[landmark_index])
        if perform_z_normalization == True:
            landmarkvector[2] = normalize_z_value(landmarkvector[2])
        ret = ret + landmarkvector

    return ret

def collect_relevant_landmarks(landmarks, relevant_landmark_indexes):
    if landmarks != None:
        return relevant_landmarks(relevant_landmark_indexes, landmarks.landmark)
    else:
        return array_zero_vectors(len(relevant_landmark_indexes))
