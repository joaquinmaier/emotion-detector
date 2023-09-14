import cv2
import numpy
import datetime
from os import getenv
import src.detector as detector
import src.recognition as recognition
import src.emotion_detector as emotion_det
import src.age_gender_detector as age_gender_detection
import json

CAMERA_INDEX = getenv('CAMERA_INDEX')

def acked(err, msg):
    if err is not None:
        print("Failed to deliver message: %s: %s" % (str(msg), str(err)))
    else:
        print("Message produced: %s" % (str(msg)))

if __name__ == "__main__":
    img = cv2.imread('./imgs/longname.jpg')
    previous_detected_faces = []
    cv2.imshow('Camera', img)

    # while True:
    detected_faces = detector.detect_faces_in_frame(img)

    utc_timestamp = datetime.datetime.now().timestamp()
    for index, face in enumerate(detected_faces):
        face = numpy.ascontiguousarray(face)
        face_locations = recognition.get_face_locations_from_image(image=face)

        if len(face_locations) == 0:
            continue

        face_encoding = recognition.get_face_encodings_from_image(image=face, face_locations=face_locations)

        emotion_detector = emotion_det.EmotionDetector()
        detected = emotion_detector.call(face)

        print(detected)

        age, gender = age_gender_detection.find_age_and_gender(face)

        print(f'Age: {age}, gender: {gender}')

        if len(previous_detected_faces) == 0:
            previous_detected_faces.append({
                        'encoding': face_encoding[0],
                        'timestamp': utc_timestamp
                    })
            
        else:
            time_threshold = 60

            is_face_in_list, already_detected = detector.face_already_detected(
                face_encoding=face_encoding,
                detected_timestamp=utc_timestamp,
                previous_detected_faces=previous_detected_faces,
                time_threshold=time_threshold
            )

            if is_face_in_list and already_detected:
                print('Face already detected less than a minute ago, omitting...')
            else:
                if is_face_in_list: print('Face already detected more than a minute ago, saving...')
                else: print('New face detected, saving...')

                previous_detected_faces.append({
                    'encoding': face_encoding[0],
                    'timestamp': utc_timestamp
                })

        cv2.imshow(f'Face detected: {index}', face)

    while True:
        k = cv2.waitKey(30) & 0xff
        if k==27:
            print('Exiting...')
            break

    cv2.destroyAllWindows()

