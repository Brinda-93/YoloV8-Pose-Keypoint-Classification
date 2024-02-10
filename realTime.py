from app import pose_classification
import cv2
import numpy as np
from PIL import Image

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    img, results_classification = pose_classification(frame)
    cv2.imshow('Pose', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
