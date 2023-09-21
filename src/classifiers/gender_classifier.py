import cv2
from src.constants import CV2_MODEL_MEAN_VALUES as MODEL_MEAN_VALUES

GENDER_PROTO = "models/gender_deploy.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"
GENDER_LIST = ['Male', 'Female']


class GenderClassifier:
    def __init__(self) -> None:
        self.gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

    def find_gender(self, frame) -> str:
        blob = cv2.dnn.blobFromImage(
            frame,
            1.0,
            (227, 227),
            MODEL_MEAN_VALUES,
            swapRB=False
        )
        self.gender_net.setInput(blob)
        gender_predictions = self.gender_net.forward()
        return GENDER_LIST[gender_predictions[0].argmax()]
