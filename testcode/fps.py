import cv2
import time

cap = cv2.VideoCapture(0)

frame_count = 0

previous_frame_time = time.time()

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

while cap.isOpened():
    ret, frame = cap.read()

    #recolor feed
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)


    #recolor image back to BGR for rendering
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    apply_fps(image)

    cv2.imshow('FPS TEST' , image)

    previous_frame_time = time.time()
    frame_count = frame_count + 1



    keyPress = cv2.waitKey(1)

    if keyPress == ord('q'):
        break
