import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

fingerCoordinates = [(8, 6), (12, 10), (16, 14), (20, 18)]
thumbCoordinates = (4, 2)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    multiLandMarks = results.multi_hand_landmarks
    
    if multiLandMarks:
        handPoints =[]
        for handLms in multiLandMarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            for idx, lm in enumerate(handLms.landmark):
                # print(idx, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                handPoints.append((cx, cy))
        
        for point in handPoints:
            cv2.circle(img, point, 10, (0, 0, 255), cv2.FILLED)    

        upCount = 0
        for coordinate in fingerCoordinates:
            if handPoints[coordinate[0]][1] < handPoints[coordinate[1]][1]:
                upCount += 1
        if handPoints[thumbCoordinates[0]][0] > handPoints[thumbCoordinates[1]][0]:
            upCount += 1

        cv2.putText(img, str(upCount), (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 0, 0), 10)

    cv2.imshow("Finger counter", img)
    cv2.waitKey(1)