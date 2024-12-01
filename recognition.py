import face_recognition
import cv2


def load_known_faces(images):
    known_faces = {}
    for name, image_path in images:
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_faces[name] = encoding
    return known_faces


def recognize_faces(frame, known_faces):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    results = []
    for face_encoding, (top, right, bottom, left) in zip(
        face_encodings, face_locations
    ):
        matches = face_recognition.compare_faces(
            list(known_faces.values()), face_encoding
        )
        name = "Sconosciuto"

        if True in matches:
            first_match_index = matches.index(True)
            name = list(known_faces.keys())[first_match_index]

        results.append((name, (top, right, bottom, left)))
    return results
