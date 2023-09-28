from paz.applications import MiniXceptionFER
import paz.processors as pr
from src.data_classes.detection_result import DetectionResult
import numpy as np

FER_EMOTION_CLASS_NAMES: list[str]  = ['angry', 'disgust', 'fear', 'happy',
                                        'sad', 'surprise', 'neutral']

class EmotionClassifier(pr.Processor):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        self.classifier = MiniXceptionFER()

    def get_max_emotion(self, image: np.array) -> DetectionResult:
        prediction = self.classifier(image)
        max_emotion = prediction['class_name']
        confidence = prediction['scores'].ravel(
        )[self.classifier.class_names.index(max_emotion)]
        return DetectionResult(max_emotion, confidence)

    def get_top_emotions(self, image, quantity=3) ->list[DetectionResult]:
        prediction_scores = self.classifier(image)['scores'].reshape(-1)

        emotions_with_scores_dict = { FER_EMOTION_CLASS_NAMES[i]: prediction_scores[i] for i in range(len(FER_EMOTION_CLASS_NAMES)) }
        emotions_with_scores = sorted(emotions_with_scores_dict.items(), key=lambda x: x[1], reverse=True)[0:quantity]

        return [ DetectionResult(emotion, confidence) for emotion, confidence in emotions_with_scores ]

