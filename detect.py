import cv2
import numpy as np
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

fer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Press SPACE to detect your mood, Q to quit")
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Mood Detector", frame)
    key = cv2.waitKey(1)

    if key == 32:
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_img = frame[y:y+h, x:x+w]
            emotion, scores = fer.predict_emotions(face_img, logits=False)
            print(f"\nDetected mood: {emotion}")
        else:
            print("No face detected!")

    elif key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()