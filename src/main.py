import cv2
import mediapipe as mp

from face_detector import create_face_landmarker, parse_landmarks, parse_blendshapes
from landmark_mapping import map_landmarks_to_blendshapes, map_mediapipe_to_vrm
from vrm_renderer import render_vrm


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Cannot access webcam.")
        return

    landmarker = create_face_landmarker()
    timestamp = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Process frame and get landmarks + blendshapes
        result = landmarker.detect_for_video(mp_image, timestamp)
        timestamp += 1

        lm_list = parse_landmarks(result)
        mp_blendshapes = parse_blendshapes(result)

        # Convert MediaPipe (ARKit) blendshapes to VRM blendshape groups
        if mp_blendshapes:
            vrm_blendshapes = map_mediapipe_to_vrm(mp_blendshapes)
            render_vrm(vrm_blendshapes)
        elif lm_list:
            # Fallback to custom mapping if blendshapes not available
            bs = map_landmarks_to_blendshapes(lm_list[0])
            render_vrm(bs)

        cv2.imshow("Face Mesh (raw)", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()