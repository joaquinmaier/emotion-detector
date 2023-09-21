from paz.applications import HaarCascadeFrontalFace
import paz.processors as pr


class FaceDetector(pr.Processor):
    def __init__(self):
        super(FaceDetector, self).__init__()
        self.face_detector = HaarCascadeFrontalFace(draw=False)

    def __call__(self, img) -> tuple():
        boxes2D = self.face_detector(img)['boxes2D']
        cropped_imgs = pr.CropBoxes2D(img, boxes2D)
        return cropped_imgs
