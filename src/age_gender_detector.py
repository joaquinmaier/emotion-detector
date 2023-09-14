#coom
import cv2

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)

face_proto="models/opencv_face_detector.pbtxt"
face_model="models/opencv_face_detector_uint8.pb"
age_proto="models/age_deploy.prototxt"
age_model="models/age_net.caffemodel"
gender_proto="models/gender_deploy.prototxt"
gender_model="models/gender_net.caffemodel"

gender_list = ['Male', 'Female']
age_list=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

def find_age_and_gender(frame) -> tuple:
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    face_net=cv2.dnn.readNet(face_model,face_proto)
    age_net=cv2.dnn.readNet(age_model, age_proto)
    gender_net=cv2.dnn.readNet(gender_model, gender_proto)

    gender_net.setInput(blob)
    gender_predictions = gender_net.forward()
    gender = gender_list[ gender_predictions[0].argmax() ]

    age_net.setInput(blob)
    age_predictions = age_net.forward()
    age = age_list[ age_predictions[0].argmax() ]

    return (age, gender)

