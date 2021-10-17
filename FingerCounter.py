import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "FingerImages"
myList = os.listdir(folderPath)
print(myList)
overlayList = &amp;#91;]
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))
pTime = 0

detector = htm.handDetector(detectionCon=0.75)

tipIds = &amp;#91;4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:
        fingers = &amp;#91;]

        # Thumb
        if lmList&amp;#91;tipIds&amp;#91;0]]&amp;#91;1] &gt; lmList&amp;#91;tipIds&amp;#91;0] - 1]&amp;#91;1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmList&amp;#91;tipIds&amp;#91;id]]&amp;#91;2] &amp;lt; lmList&amp;#91;tipIds&amp;#91;id] - 2]&amp;#91;2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w, c = overlayList&amp;#91;totalFingers - 1].shape
        img&amp;#91;0:h, 0:w] = overlayList&amp;#91;totalFingers - 1]

        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)