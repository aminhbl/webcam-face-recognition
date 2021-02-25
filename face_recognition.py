import numpy as np
import cv2 as cv
import time

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

people = ['amin', 'Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')


def recognise(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow('Person', gray)

    # Detect the face in the image
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y + h, x:x + w]

        label, confidence = face_recognizer.predict(faces_roi)
        print(f'Label = {people[label]} with a confidence of {confidence}')

        cv.putText(img, str(people[label]), (x, y - 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), thickness=1)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=1)

    cv.imshow('Detected Face', img)


capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    recognise(frame)
    # time.sleep(0.01)
    if cv.waitKey(1) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
