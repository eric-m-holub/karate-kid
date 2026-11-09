#feature detection thresholds
min_detection_confidence = 0.8
min_tracking_confidence = 0.8


# of labels for model
actions = ['NO ACTION', 'SAND THE FLOOR', 'WAX ON / WAX OFF', 'PAINT THE FENCE', 'PAINT THE HOUSE']

# indexes of landmarks we want to save for hands and poses
relevant_hand_landmark_indexes = [0,4,8,12,16,20]
relevant_pose_landmark_indexes = [15,13,11,16,14,12]


# how many frames will be processed at a time by our model
frames_per_data_set = 14

#number of frames which should be skipped when collecting / inferring data
frame_skip = 1

#should and normalize z values and what is the absolute max z value for our normalization calculation
perform_z_normalization = True
z_normalization_absolute_max = 1


#
cutoff_for_best_prediction = 0.51
