import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = []

            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            for tip in finger_tips:
                x, y = int(lm_list[tip].x*w), int(lm_list[tip].y*h)
                # print(id, ":", x, y)
                cv2.circle(img, (x,y), 15, (255, 0, 0), cv2.FILLED)

            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracking", img)
    cv2.waitKey(1)