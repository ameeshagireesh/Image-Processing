import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]
thumb_tip = 4

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
            finger_fold_status = []
            
            for tip in finger_tips:
                x, y = int(lm_list[tip].x*w), int(lm_list[tip].y*h)
                # print(id, ":", x, y)
                cv2.circle(img, (x,y), 15, (255, 0, 0), cv2.FILLED)

                if lm_list[tip].x < lm_list[tip-3].x:
                    cv2.circle(img, (x,y), 15, (0, 255, 0), cv2.FILLED)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)
            
            print(finger_fold_status)

            if all(finger_fold_status):
                if lm_list[thumb_tip].y < lm_list[thumb_tip - 1].y < lm_list[thumb_tip-2].y:
                    print("LIKE")
                    cv2.putText(img, "LIKE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                if lm_list[thumb_tip].y > lm_list[thumb_tip - 1].y > lm_list[thumb_tip-2].y:
                    print("DISLIKE")
                    cv2.putText(img, "DISLIKE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracking", img)
    cv2.waitKey(1)