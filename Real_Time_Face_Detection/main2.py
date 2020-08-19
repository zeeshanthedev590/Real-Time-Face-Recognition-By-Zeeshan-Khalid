import cv2
from random import randrange
# Loading The Cascade File
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('eye.xml')
glass_cascade = cv2.CascadeClassifier('glass.xml')
# smile_cascade = cv2.CascadeClassifier('smile.xml')
webcam = cv2.VideoCapture(0)
while True:
    successfull_frame_read, frame = webcam.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_tracker = face_cascade.detectMultiScale(gray_img)
    # smile_tracker = smile_cascade.detectMultiScale(gray_img)
    glass_tracker = glass_cascade.detectMultiScale(gray_img)
    t = len(face_tracker)
    cv2.putText(frame,  str(t), (40, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 77, 255), 2, cv2.LINE_AA)
    for (x, y, w, h) in face_tracker:
        font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
        org = (x, y-40)
        fontScale = 2
        color = (0, 255, 0)
        thickness = 3
        cap = cv2.putText(frame, 'Face', org, font,
                          fontScale, color, thickness, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (randrange(256), randrange(256), randrange(256)), 10)
        print("Face Detected = ", len(face_tracker))
        for (ex, ey, ew, eh) in glass_tracker:
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh),
                          (0, 0, 0), 5)

    cv2.imshow('Face Detector by Zeeshan Khalid', frame)
    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('c') or key == ord('d') or key == ord('e'):
        break
