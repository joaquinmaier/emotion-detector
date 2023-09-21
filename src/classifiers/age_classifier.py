import cv2
from src.constants import CV2_MODEL_MEAN_VALUES as MODEL_MEAN_VALUES

AGE_PROTO = "models/age_deploy.prototxt"
AGE_MODEL = "models/age_net.caffemodel"
AGE_RANGES_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                   '(25-32)', '(38-43)', '(48-53)', '(60-100)']


class AgeClassifier:
    def __init__(self) -> None:
        self.age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)

    def _get_age_predictions(self, frame):
        blob = cv2.dnn.blobFromImage(
            frame,
            1.0,
            (227, 227),
            MODEL_MEAN_VALUES,
            swapRB=False
        )
        self.age_net.setInput(blob)
        age_predictions = self.age_net.forward()
        return age_predictions

    def max_age_range(self, frame) -> str:
        age_predictions = self._get_age_predictions(frame)
        return AGE_RANGES_LIST[age_predictions[0].argmax()]

    def top_age_ranges(self, frame, amount=3) -> tuple:
        age_predictions = self._get_age_predictions(frame)
        return AGE_RANGES_LIST[age_predictions[0].argsort()[::-1]][:amount]
