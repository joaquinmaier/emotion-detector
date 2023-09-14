import face_recognition

MODEL = 'hog'

def get_face_locations_from_image(image):
    """
    Returns the face locations for the faces in the given image.
    """
    return face_recognition.face_locations(img=image, model=MODEL, number_of_times_to_upsample=2)

def get_face_encodings_from_image(image, face_locations=None):
    """
    Returns the face encodings for the faces in the given image.
    """
    return face_recognition.face_encodings(
        face_image=image,
        known_face_locations=face_locations
    )

def compare_faces(face_encodings, face_to_compare):
    """
    Compares the given face encodings and returns True if they match
    """
    return face_recognition.compare_faces(face_encodings, face_to_compare)