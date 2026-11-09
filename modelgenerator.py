import keras
from keras.models import Sequential, load_model
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras import regularizers

import json
import os


from constants import frames_per_data_set
import logichelper


collectedData = []


with open('data/karate-kid.json') as json_file:
    print(json_file)
    json_data = json.load(json_file)
    collectedData = json_data


train_samples = []
train_labels = []
output_count = len(collectedData)


for label_index in range(output_count):                                             #parse each label
    training_sessions = collectedData[label_index]

    for training_session_index in range(len(training_sessions)):                           #parse each training session from each label
        training_session = training_sessions[training_session_index]

        if len(training_session) >= frames_per_data_set:                            #training session must have a least frames_per_data_set number of frames

            frames_for_session = training_session
            frame_count_for_session = len(frames_for_session)

            for starting_frame_index in range(frame_count_for_session-(frames_per_data_set-1)):
                first_frame = frames_for_session[starting_frame_index]
                frame_sequence_array = [] + first_frame
                for previous_frame_index in range(starting_frame_index+1, starting_frame_index+frames_per_data_set):
                    previous_frame = frames_for_session[previous_frame_index]
                    frame_sequence_array = frame_sequence_array + previous_frame

                train_samples.append(frame_sequence_array)
                train_labels.append(label_index)

dimensions = logichelper.data_input_dimensions()



if os.path.exists('models/karate-kid.h5'):
    model = load_model('models/karate-kid.h5')
else:
    model = Sequential([
        Dense(dimensions, input_shape=(dimensions,), activation='relu'),
        Dense(dimensions*2, input_shape=(dimensions,), activation='relu'),
        Dense(dimensions*4, input_shape=(dimensions,), activation='relu'),
        Dense(output_count, activation='softmax')
    ])

model.compile(Adam(lr=.0001), loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.fit(x=train_samples, y=train_labels, batch_size=20, epochs=10, shuffle=True, verbose=2)


model.save('models/karate-kid.h5')
