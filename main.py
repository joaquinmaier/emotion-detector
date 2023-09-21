import os
import sys
import numpy
import cv2
from src.face_detector import FaceDetector
from src.classifiers.age_classifier import AgeClassifier
from src.classifiers.gender_classifier import GenderClassifier
from src.classifiers.emotion_classifier import EmotionClassifier

IMGS_PATH = os.path.join(os.path.dirname(__file__), 'imgs')
TEST_IMGS_PATH = os.path.join(IMGS_PATH, 'tests')
CURRENT_IMGS_PATH = os.path.join(IMGS_PATH, 'current')
ESC_KEYCODE = 27
ENTER_KEYCODE = 13


def main():
    images_path = CURRENT_IMGS_PATH
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        images_path = TEST_IMGS_PATH

    images = os.listdir(images_path)
    if len(images) == 0:
        raise FileNotFoundError(
            "No images found in the 'imgs/current' folder. Please add some images to the folder and try again or run the program with --test for testing.")

    age_classifier = AgeClassifier()
    gender_classifier = GenderClassifier()
    emotion_classifier = EmotionClassifier()
    face_detector = FaceDetector()
    interrupted = False

    for image_path in images:
        if interrupted:
            break

        img = cv2.imread(os.path.join(images_path, image_path))
        detected_faces = face_detector(img=img)
        print(detected_faces)
        cv2.imshow('Original Image', img)

        '''
        for index, face in enumerate(detected_faces):
            face = numpy.ascontiguousarray(face)
            face_locations = recognition.get_face_locations_from_image(
                image=face)

            if len(face_locations) == 0:
                continue

            emotion_det = emotion_detector.EmotionDetector()
            detected = emotion_det.get_max_emotion_from_faces(face)

            age, gender = gender_detector.find_age_and_gender(face)

            print(f'Age: {age}, gender: {gender}')
            cv2.imshow(f'Face detected: {index}', face)
        '''
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
