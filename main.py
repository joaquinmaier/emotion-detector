import os
import sys
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

    print("Initializing...")

    age_classifier = AgeClassifier()
    gender_classifier = GenderClassifier()
    emotion_classifier = EmotionClassifier()
    face_detector = FaceDetector()
    interrupted = False

    print(f"Found {len(images)} images in {images_path}")

    for image_path in images:
        if interrupted:
            break

        img = cv2.imread(os.path.join(images_path, image_path))
        detected_faces = face_detector(img=img)

        if len(detected_faces) == 0:
            print("No faces detected in the image. Skipping...")
            continue

        cv2.imshow('Original Image', img)

        for index, face in enumerate(detected_faces):
            predicted_emotions = emotion_classifier.get_top_emotions(face)
            age = age_classifier.max_age_range(face)
            gender = gender_classifier.find_gender(face)

            print(f'Age: {age}, gender: {gender}')
            print(f'-- Predicted Emotion (Model\'s choice) --\n\tEmotion: {predicted_emotions[0].result} - Confidence: {predicted_emotions[0].confidence}')

            print('\n-- Other possible emotions --\n')
            for i in range(1, len(predicted_emotions)):
                print(f'({i})\tEmotion: {predicted_emotions[i].result} - Confidence: {predicted_emotions[i].confidence}')

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
