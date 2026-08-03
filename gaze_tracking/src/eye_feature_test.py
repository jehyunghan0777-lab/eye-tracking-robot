from __future__ import annotations

from pathlib import Path
import sys
import time

import cv2
import mediapipe as mp

WINDOW_NAME = "Eye-Tracking Robot - Eye Features"

# __file__is this Python file.
# parents[0] is src/
# parents[1] is gaze_tracking/
MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "face_landmarker.task"
)

# Eye-counter landmarks defined by MediaPipe. 
RIGHT_EYE_INDICES = {
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 246, 161, 160, 159, 158, 157, 173,
}

LEFT_EYE_INDICES = {
    263, 249, 390, 373, 374, 380, 381, 382,
    362, 466, 388, 387, 386, 385, 384, 398,
}

# Iris landmarks defined by MediaPipe.
RIGHT_IRIS_INDICES = {468, 469, 470, 471, 472}
LEFT_IRIS_INDICES = {473, 474, 475, 476, 477}

# Iris-center landmarks.
RIGHT_IRIS_CENTER_INDEX = 468
LEFT_IRIS_CENTER_INDEX = 473

# Horizontal eye corners.
RIGHT_EYE_CORNER_INDEX = (33, 133)
LEFT_EYE_CORNER_INDEX = (263, 362)  

# Approximate upper and lower eyelid landmarks.
RIGHT_EYELID_VERTICAL_INDICES = (159, 145)
LEFT_EYELID_VERTICAL_INDICES = (386, 374)


def open_camera() -> cv2.VideoCapture:
    """Open the default Windows webcam."""

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if camera.isOpened():
        print("Camera opened using DirectShow. ")
        return camera

    print("Directshow failed. Trying automatic backend.")
    camera.release()

    return cv2.VideoCapture(0)

def draw_eye_features(
    frame,
    face_landmarks,
) -> None:
    """Draw only eye-contour and iris landmarks."""

    frame_height, frame_width = frame.shape[:2]

    for landmark_index, landmark in enumerate(face_landmarks):
        is_eye_landmark = (
            landmark_index in RIGHT_EYE_INDICES
            or landmark_index in LEFT_EYE_INDICES
        )

        is_iris_landmark = (
            landmark_index in RIGHT_IRIS_INDICES
            or landmark_index in LEFT_IRIS_INDICES
        )

        if not is_eye_landmark and not is_iris_landmark:
            continue

        pixel_x = int(landmark.x * frame_width)
        pixel_y = int(landmark.y * frame_height)

        pixel_x = max(0, min(pixel_x, frame_width - 1))
        pixel_y = max(0, min(pixel_y, frame_height - 1))

        if is_iris_landmark:
            color = (0, 0, 255)  # Red for iris landmarks.
            radius = 3
        else:
            color = (0, 255, 0)  # Green for eye-contour landmarks.
            radius = 2
        
        cv2.circle(
            frame, 
            (pixel_x, pixel_y), 
            radius, 
            color, 
            -1,
            cv2.LINE_AA,    
        )

def calculate_eye_position(
    face_landmarks, 
    iris_center_index,
    eye_corner_indices,
    eye_vertical_indices,
):
    """Calculate the normalized iris position inside one eye."""

    iris_center = face_landmarks[iris_center_index]

    corner_a = face_landmarks[eye_corner_indices[0]]
    corner_b = face_landmarks[eye_corner_indices[1]]

    upper_eyelid = face_landmarks[eye_vertical_indices[0]]
    lower_eyelid = face_landmarks[eye_vertical_indices[1]]

    # Determine the horizontal eye boundries.
    eye_left_x = min(corner_a.x, corner_b.x)
    eye_right_x = max(corner_a.x, corner_b.x)

    # Determine the vertical eye boundries.
    eye_top_y = min(upper_eyelid.y, lower_eyelid.y)
    eye_bottom_y = max(upper_eyelid.y, lower_eyelid.y)

    eye_width = eye_right_x - eye_left_x
    eye_height = eye_bottom_y - eye_top_y

    # Protect against division by zero
    if eye_width <= 0 or eye_height <= 0:
        return None

    horizontal_ratio = (iris_center.x - eye_left_x) / eye_width

    vertical_ratio = (iris_center.y - eye_top_y) / eye_height

    return horizontal_ratio, vertical_ratio
    

def main() -> int:
    """Run real-time MediPipe facial-landmark detection."""

    if not MODEL_PATH.exists():
        print(f"Error: Model file not found at {MODEL_PATH}",
        file=sys.stderr,
        )
        return 1

    print(f"Using model: {MODEL_PATH}")

    camera = open_camera()

    if not camera.isOpened():
        print("ERROR: OpenCV could not open the webcame.",
        file=sys.stderr,
        )
        return 1

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # These aliases make the MediaPipe setup easier to read. 
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH),
        ),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    start_time = time.perf_counter()

    try:
        with FaceLandmarker.create_from_options(options) as landmarker:
            print("Face Landmarker intialized.")
            print("Press Q or ESC to quit.")

            while True:
                frame_received, frame = camera.read()

                if not frame_received or frame is None:
                    print(
                        "ERROR: failed to receive a webcame frame.",
                        file=sys.stderr,
                    )
                    return 1

                # Mirror the webcame preview. 
                frame = cv2.flip(frame, 1)

                # OpenCV stores images as BGR.
                # MediaPipe expects RGB.
                rgb_frame = cv2.cvtColor(
                    frame, 
                    cv2.COLOR_BGR2RGB
                )

                mediapipe_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                # Video mode requires a timestamp that always increases.
                timestamp_ms = int(
                    (time.perf_counter() - start_time) * 1000
                )

                result = landmarker.detect_for_video(
                    mediapipe_image,
                    timestamp_ms,
                )

                if result.face_landmarks:
                    face_landmarks = result.face_landmarks[0]

                    draw_eye_features(
                        frame, 
                        face_landmarks,
                    )

                    right_eye_position = calculate_eye_position(
                        face_landmarks,
                        RIGHT_IRIS_CENTER_INDEX,
                        RIGHT_EYE_CORNER_INDEX,
                        RIGHT_EYELID_VERTICAL_INDICES,
                    )

                    left_eye_position = calculate_eye_position(
                        face_landmarks,
                        LEFT_IRIS_CENTER_INDEX,
                        LEFT_EYE_CORNER_INDEX,
                        LEFT_EYELID_VERTICAL_INDICES,
                    )

                    if right_eye_position is not None and left_eye_position is not None:
                        right_horizontal, right_vertical = right_eye_position
                        left_horizontal, left_vertical = left_eye_position

                        average_horizontal = (
                            right_horizontal + left_horizontal
                        ) / 2.0

                        average_vertical = (
                            right_vertical + left_vertical
                        ) / 2.0

            

                    status_text = (
                        f"Eye position: |"
                        f"H: {average_horizontal:.2f} "
                        f"V: {average_vertical:.2f}"
                        )

                else:
                    status_text = "No face detected"

                cv2.putText(
                    frame,
                    status_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame, 
                    "Green: eye contour | Red: iris | Q/ESC: exit",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:  # ESC
                    print("Exiting...")
                    break
    
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("Eye features test closed cleanly.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
