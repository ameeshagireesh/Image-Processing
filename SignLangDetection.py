import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]

while True:
    ret, img = cap.read()
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = []

            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append()

            for tip in finger_tips:
                x, y = int(lm_list.x*w), int(lm.y*h)
                print(id, ":", x, y)

                if id==8:
                    cv2.circle(img, (x,y), 15, (255, 0, 0), cv2.FILLED)

            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracking", img)
    cv2.waitKey(1)