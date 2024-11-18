import os.path

import numpy as np
from sympy import pprint
from ultralytics import YOLO
import cv2
import math
import matplotlib.pyplot as plt

from KalmanFilter import WalkingPedestrianKalmanFilter

# start webcam
cap = cv2.VideoCapture("videos/walking_cut.mp4")
#cap.set(3, 640)
#cap.set(4, 480)

# model
model = YOLO("yolo_weights/yolo11n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

timestamps = []
positions_measured = []
positions_predicted = []
positions_corrected = []

kalman_filter = None
kalman_initialized = False

success = True
while success:
    success, img = cap.read()


    if success:
        results = model(img, stream=True)
        already_seen = False
        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                _x1, _y1, _x2, _y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(_x1), int(_y1), int(_x2), int(_y2) # convert to int values

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # Detect the person
                if classNames[cls] == "person" and not already_seen:
                    timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
                    x_pos = (_x1 + _x2)/2
                    y_pos = (_y1 + _y2)/2
                    positions_measured.append((x_pos, y_pos))
                    already_seen = True # Make sure that only one pedestrian has been registered at each step

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

        # Kalman filter Initilization (use first 2 loops to make a first estimation of speed)
        if kalman_filter is None:
            kalman_filter = WalkingPedestrianKalmanFilter(
                1/cap.get(cv2.CAP_PROP_FPS),
                5,
                15,
                0.95
            )

        elif not kalman_initialized and len(positions_measured) >= 2 :
            # speed estimation
            X0 = np.array(positions_measured[-1])
            X1 = np.array(positions_measured[-2])

            t0 = timestamps[-1]
            t1 = timestamps[-2]

            dX = X1 - X0
            dt = t1 - t0

            V = dX/dt

            p = 0.6

            kalman_filter.initialize(X0[0], X0[1], V[0], V[1], p)
            X_predicted = kalman_filter.predict() # Predict next state
            positions_corrected += [np.array([[0], [0], [0], [0]])] * len(positions_measured)
            positions_predicted += [np.array([[0], [0], [0], [0]])] * len(positions_measured)
            positions_predicted.append(X_predicted)
            kalman_initialized = True

        elif kalman_initialized:

            Zmeasure = np.array([
                [positions_measured[-1][0]],
                [positions_measured[-1][1]]
            ])

            X_corrected = kalman_filter.correct(Zmeasure)
            positions_corrected.append(X_corrected)

            X_predicted = kalman_filter.predict()
            positions_predicted.append(X_predicted)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()



#plot
positions_measured = np.array(positions_measured)
X_measured = positions_measured[:, 0]
Y_measured = positions_measured[:, 1]
plt.plot(X_measured, Y_measured)

positions_predicted = np.array(positions_predicted)
X_predicted = positions_predicted[5:, 0]
Y_predicted = positions_predicted[5:, 1]
plt.plot(X_predicted, Y_predicted)

ax = plt.gca()
ax.set_xlim([0, 900])
ax.set_ylim([0, 500])

save_path = "out/trajectory_measured.png"
if os.path.exists(save_path):
    os.remove(save_path)
plt.savefig(save_path)
