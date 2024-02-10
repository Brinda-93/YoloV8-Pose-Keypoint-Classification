# Import library
import cv2
import glob
import numpy as np
from PIL import Image
# import streamlit as st

from src.detection_keypoint import DetectKeypoint
from src.classification_keypoint import KeypointClassification

detection_keypoint = DetectKeypoint()
classification_keypoint = KeypointClassification(
    './models/pose_classification.pth'
)

def pose_classification(img, col=None):
    image_cv = img

    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    # detection keypoint
    results = detection_keypoint(image_cv)
    results_keypoint = detection_keypoint.get_xy_keypoint(results)

    # classification keypoint
    input_classification = results_keypoint[10:]
    results_classification = classification_keypoint(input_classification)

    # visualize result
    image_draw = results.plot(boxes=False)
    x_min, y_min, x_max, y_max = results.boxes.xyxy[0].numpy()
    image_draw = cv2.rectangle(
                    image_draw,
                    (int(x_min), int(y_min)),(int(x_max), int(y_max)),
                    (0,0,255), 2
                )
    (w, h), _ = cv2.getTextSize(
                    results_classification.upper(),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
    image_draw = cv2.rectangle(
                    image_draw,
                    (int(x_min), int(y_min)-20),(int(x_min)+w, int(y_min)),
                    (0,0,255), -1
                )
    image_draw = cv2.putText(image_draw,
                    f'{results_classification.upper()}',
                    (int(x_min), int(y_min)-4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255),
                    thickness=2
                )
    image_draw = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)
    return image_draw, results_classification
