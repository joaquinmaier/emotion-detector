import face_recognition

MODEL = 'hog'
UPSAMPLE_TIMES = 2

def get_face_locations_from_image(image):
    return face_recognition.face_locations(img=image, model=MODEL, number_of_times_to_upsample=UPSAMPLE_TIMES)

def get_face_encodings_from_image(image, face_locations=None):
    return face_recognition.face_encodings(
        face_image=image,
        known_face_locations=face_locations
    )

def compare_faces(face_encodings, face_to_compare):
    return face_recognition.compare_faces(face_encodings, face_to_compare)