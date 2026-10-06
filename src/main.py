import cv2
import mediapipe as mp

from landmark_mapping import map_landmarks_to_blendshapes
from vrm_renderer import render_vrm


mp_face_mesh = mp.solutions.face_mesh
FACE_DETECTOR = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
)

def parse_landmarks(results):
    out = []
    if results.multi_face_landmarks:
        for fm in results.multi_face_landmarks:
            face = {f"lm{i}": (lm.x, lm.y, lm.z) for i, lm in enumerate(fm.landmark)}
            out.append(face)
    return out


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Cannot access webcam.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = FACE_DETECTOR.process(rgb)
        lm_list = parse_landmarks(results)

        if lm_list:
            bs = map_landmarks_to_blendshapes(lm_list[0])
            render_vrm(bs)

        cv2.imshow("Face Mesh (raw)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()