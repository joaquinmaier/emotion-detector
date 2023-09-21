import os
import sys
import cv2
import numpy
import src.detector as detector
import src.recognition as recognition
import src.emotion_detector as emotion_detector
import src.age_gender_detector as age_gender_detector

IMGS_PATH           = os.path.join(os.path.dirname(__file__), 'imgs')
TEST_IMGS_PATH      = os.path.join(IMGS_PATH, 'tests')
CURRENT_IMGS_PATH   = os.path.join(IMGS_PATH, 'current')
ESC_KEYCODE         = 27
ENTER_KEYCODE       = 13


def main():
    images_path = CURRENT_IMGS_PATH

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        images_path = TEST_IMGS_PATH

    images = os.listdir(images_path)
    interrupted = False

    for image_path in images:
        if interrupted:
            break

        img = cv2.imread(os.path.join(images_path, image_path))

        cv2.imshow('Original Image', img)

        detected_faces = detector.detect_faces_in_frame(img)

        for index, face in enumerate(detected_faces):
            face = numpy.ascontiguousarray(face)
            face_locations = recognition.get_face_locations_from_image(image=face)

            if len(face_locations) == 0:
                continue

            emotion_det = emotion_detector.EmotionDetector()
            detected = emotion_det.call(face)

            age, gender = age_gender_detector.find_age_and_gender(face)

            print(f'Age: {age}, gender: {gender}')
            cv2.imshow(f'Face detected: {index}', face)

        while True:
            k = cv2.waitKey(30) & 0xff
            if k == ESC_KEYCODE:
                interrupted = True
                break

            elif k == ENTER_KEYCODE:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
