import cv2
import uuid
import src.editor as editor
import src.recognition as recognition


def detect_faces_in_frame(frame):
    face_cascade = cv2.CascadeClassifier(
        'models/haarcascade_frontalface_default.xml')

    # Convert into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
    detected_faces = []

    for (x, y, w, h) in faces:
        roi_image = editor.crop_region_of_interest(
            frame,
            x=x,
            y=y,
            w=w,
            h=h,
            extra_size=50,
            show_image=False
        )

        if roi_image is not None:
            detected_faces.append(roi_image)

    return detected_faces


def start_detecting_faces_from_video(video_path):
    """
    Detects faces from the camera and returns the image with bounding boxes
    around the faces.
    """
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(
        'models/haarcascade_frontalface_default.xml')
    face_id = uuid.uuid4()

    # To capture video from webcam.
    cap = cv2.VideoCapture(video_path)

    while True:
        # Read the frame
        _, img = cap.read()

        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))

        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            rectangle_extra_size = 50

            # Draw the rectangle
            # cv2.rectangle(img, (x-rectangle_extra_size, y-rectangle_extra_size), (x+w+rectangle_extra_size, y+h+rectangle_extra_size), (255, 0, 0), 2)

            # Crop the face with extra size
            roi_color = img[y-rectangle_extra_size:y+h+rectangle_extra_size,
                            x-rectangle_extra_size:x+w+rectangle_extra_size]

            if roi_color.shape[0] > 0 and roi_color.shape[1] > 0:
                cv2.imshow('face', roi_color)
                # Save the cropped face
                cv2.imwrite(
                    './cropped_faces/face{}.jpg'.format(face_id), roi_color)
            else:
                # Crop the face without extra size
                roi_color = img[y:y+h, x:x+w]
                if roi_color.shape[0] > 0 and roi_color.shape[1] > 0:
                    cv2.imshow('face', roi_color)
                    # Save the cropped face
                    cv2.imwrite(
                        './cropped_faces/face{}.jpg'.format(face_id), roi_color)

            # Generate random uuid
            face_id = uuid.uuid4()

        # Display the output
        cv2.imshow('img', img)

        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    # Release the VideoCapture object
    cap.release()


def face_already_detected(
    face_encoding,
    detected_timestamp: float,
    previous_detected_faces: list = [],
    time_threshold: int = 60
) -> list:
    if len(previous_detected_faces) == 0:
        return [False, False]

    result: list = [False, False]

    for previous_face in previous_detected_faces.__reversed__():
        if recognition.compare_faces(
            face_encoding,
            previous_face.get('encoding')
        )[0]:
            result[0] = True

            if detected_timestamp - previous_face.get('timestamp') < time_threshold:
                result[1] = True
            break
    return result
