import cv2 as cv
import os
import face_training

faces = []
counter = 0

capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)
    counter += 1
    if counter % 2 == 0:
        faces.append(frame)

    if cv.waitKey(1) & 0xFF == ord('d'):
        break

# save pictures
print('Counter =', counter)
for i in range(0, len(faces)):
    cv.imwrite('photos/amin/{}.jpg'.format(i), faces[i])


print("Saved!")
face_training.create_train()
capture.release()
cv.destroyAllWindows()