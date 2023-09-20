# coom
import cv2

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

FACE_PROTO = "models/opencv_face_detector.pbtxt"
FACE_MODEL = "models/opencv_face_detector_uint8.pb"
AGE_PROTO = "models/age_deploy.prototxt"
AGE_MODEL = "models/age_net.caffemodel"
GENDER_PROTO = "models/gender_deploy.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"

GENDER_LIST = ['Male', 'Female']
AGE_RANGES_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                   '(25-32)', '(38-43)', '(48-53)', '(60-100)']


def find_age_and_gender(frame) -> tuple:
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
    gender_net.setInput(blob)
    gender_predictions = gender_net.forward()
    gender = GENDER_LIST[gender_predictions[0].argmax()]

    age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
    age_net.setInput(blob)
    age_predictions = age_net.forward()
    age = AGE_RANGES_LIST[age_predictions[0].argmax()]

    return (age, gender)
