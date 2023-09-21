from paz.applications import MiniXceptionFER
import paz.processors as pr
from src.detection_result import DetectionResult
import numpy as np


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

    def get_top_emotions(self, image, amount=3):  # -> tuple(DetectionResult):
        raise NotImplementedError("Not implemented")

    '''
      scores = sorted([(score, emotion) for emotion, score in zip(
                self.classifier.class_names, prediction['scores'].ravel())], reverse=True)

            print(klas)
            print(score)
            print(scores)

            results.append({'face': cropped_image,
                            'emotion': klas,
                            'scores': scores})
          
    '''
