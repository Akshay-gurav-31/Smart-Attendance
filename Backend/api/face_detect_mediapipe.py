import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection

def detect_faces_mediapipe(img_array, min_confidence=0.5):
    cropped_faces = []
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=min_confidence) as face_detection:
        results = face_detection.process(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return []

        h, w, _ = img_array.shape
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x1 = int(bboxC.xmin * w)
            y1 = int(bboxC.ymin * h)
            x2 = int((bboxC.xmin + bboxC.width) * w)
            y2 = int((bboxC.ymin + bboxC.height) * h)

            face = img_array[y1:y2, x1:x2]
            if face.shape[0] < 50 or face.shape[1] < 50:
                continue
            face = cv2.resize(face, (112, 112))
            cropped_faces.append(face)

    return cropped_faces
