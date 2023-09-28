import os
import sys
import cv2
from deepface import DeepFace
from src.face_detector import FaceDetector

IMGS_PATH = os.path.join(os.path.dirname(__file__), 'imgs')
TEST_IMGS_PATH = os.path.join(IMGS_PATH, 'tests')
CURRENT_IMGS_PATH = os.path.join(IMGS_PATH, 'current')
ESC_KEYCODE = 27
ENTER_KEYCODE = 13
EXTRA_INFO_TOP_QUANTITY = 3

def main():
    cmd_args = sys.argv[1:] if len(sys.argv) > 1 else None

    images_path = CURRENT_IMGS_PATH
    if cmd_args is not None:
        for arg in cmd_args:
            match arg:
                case "--test":
                    images_path = TEST_IMGS_PATH

    images = os.listdir(images_path)
    if len(images) == 0:
        raise FileNotFoundError(
            "No images found in the 'imgs/current' folder. Please add some images to the folder and try again or run the program with --test for testing.")

    print("Initializing...")

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
            rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            predictions: dict = {}
            try:
                predictions = DeepFace.analyze(rgb_face)[0]

            except:
                print('\x1b[1;31mDeepFace detected no faces.\x1b[0m')
                break

            print(f'\n-- Results [{index}] --')
            print(f'Age\t->\t{predictions["age"]}')
            print(f'Gender\t->\t{predictions["dominant_gender"]} (confidence: {round(predictions["gender"][ predictions["dominant_gender"] ], 3)}%)')
            print(f'Emotion\t->\t{predictions["dominant_emotion"]} (confidence: {round(predictions["emotion"][ predictions["dominant_emotion"] ], 3)}%)')
            print(f'Race\t->\t{predictions["dominant_race"]} (confidence: {round(predictions["race"][ predictions["dominant_race"] ], 3)}%)')

            print('\n\n++++++++++')
            print(f'Top {EXTRA_INFO_TOP_QUANTITY} emotions by highest confidence:')

            emotions_sorted_by_confidence = sorted(
                predictions["emotion"].items(),
                key=lambda x: x[1],
                reverse=True
            )[0:EXTRA_INFO_TOP_QUANTITY]

            for i in range(len(emotions_sorted_by_confidence)):
                print(f'({i})\t{emotions_sorted_by_confidence[i][0]} (confidence: {round(emotions_sorted_by_confidence[i][1], 3)}%)')

            print(f'\nTop {EXTRA_INFO_TOP_QUANTITY} races by highest confidence:')

            races_sorted_by_confidence = sorted(
                predictions["race"].items(),
                key=lambda x: x[1],
                reverse=True
            )[0:EXTRA_INFO_TOP_QUANTITY]

            for i in range(len(races_sorted_by_confidence)):
                print(f'({i})\t{races_sorted_by_confidence[i][0]} (confidence: {round(races_sorted_by_confidence[i][1], 3)}%)')

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

