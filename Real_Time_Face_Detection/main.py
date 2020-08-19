import cv2
from random import randrange
# Loading The Cascade File
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('eye.xml')
glass_cascade = cv2.CascadeClassifier('glass.xml')
webcam = cv2.VideoCapture(0)
while True:
    successfull_frame_read, frame = webcam.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_tracker = face_cascade.detectMultiScale(gray_img)
    # eye_tracker = eye_cascade.detectMultiScale(gray_img)
    glass_tracker = glass_cascade.detectMultiScale(gray_img)

    for (x, y, w, h) in face_tracker:
        font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
        org = (x, y-40)
        fontScale = 2
        color = (0, 255, 0)
        thickness = 5
        cap = cv2.putText(frame, 'Face', org, font,
                          fontScale, color, thickness, cv2.LINE_AA)
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h),
                      (randrange(256), randrange(256), randrange(256)), 10)
    for (x, y, w, h) in glass_tracker:
        cv2.rectangle(frame, (x+4, y+2), (x+w, y+h),
                      (0, 0, 0), 5)
    # for (x, y, w, h) in eye_tracker:
    #     cv2.rectangle(frame, (x+1, y+2), (x+w, y+h),
    #                   (255, 255, 255), 5)
    cv2.imshow('Face Detector by Zeeshan Khalid', frame)
    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('c') or key == ord('d') or key == ord('e'):
        break
