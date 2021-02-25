import os
import cv2 as cv
import numpy as np

DIR = r'photos'

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
user = ['amin', 'Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

features = []
labels = []


def create_train():
    for person in user:
        path = os.path.join(DIR, person)
        label = user.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            if img_array is None:
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y + h, x:x + w]
                features.append(faces_roi)
                labels.append(label)
    after_train()


def after_train():
    global features
    global labels

    features = np.array(features, dtype='object')
    labels = np.array(labels)

    face_recognizer = cv.face.LBPHFaceRecognizer_create()

    # Train the Recognizer on the features list and the labels list
    face_recognizer.train(features, labels)
    print('Training done ---------------')

    face_recognizer.save('face_trained.yml')
    np.save('features.npy', features)
    np.save('labels.npy', labels)
