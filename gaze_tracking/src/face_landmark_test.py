from __future__ import annotations

from pathlib import Path
import sys
import time

import cv2
import mediapipe as mp

WINDOW_NAME = "Eye-Tracking Robot - MediaPipe Face Landmarks"

# __file__is this Python file.
# parents[0] is src/
# parents[1] is gaze_tracking/
MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "face_landmarker.task"
)

def open_camera() -> cv2.VideoCapture:
    """Open the default Windows webcam."""

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if camera.isOpened():
        print("Camera opened using DirectShow. ")
        return camera

    print("Directshow failed. Trying automatic backend.")
    camera.release()

    return cv2.VideoCapture(0)

def draw_face_landmarks(
    frame,
    face_landmarks,
) -> None:

    """Draw facial landmarks onto an OpenCV frame."""

    frame_height, frame_width = frame.shape[:2]

    for landmark_index, landmark in enumerate(face_landmarks):
        # MediaPipe returns normalized x and y coordinates.
        # Multiplying by the image dimensions coverts them to pixels. 
        pixel_x = int(landmark.x * frame_width)
        pixel_y = int(landmark.y * frame_height)

        # Keep coordinates inside the image. 
        pixel_x = max(0, min(pixel_x, frame_width - 1))
        pixel_y = max(0, min(pixel_y, frame_height - 1))

        # Landmarks 468-477 are the ten iris-related landmarks. 
        if 468 <= landmark_index <= 477:
            color = (0, 0, 255)  # Red for iris landmarks.
            radius = 3
        else:
            color = (0, 255, 0)  # Green for other landmarks.
            radius = 1

        cv2.circle(
            frame, 
            (pixel_x, pixel_y), 
            radius, 
            color, 
            -1,
            cv2.LINE_AA,
        )

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

                    draw_face_landmarks(
                        frame, 
                        face_landmarks,
                    )

                    status_text = (
                        f"Face detected | "
                        f"Landmarks: {len(face_landmarks)}"
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
                    "Green: face | Red: iris | Q/ESC: exit",
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

    print("Face-landmakr test closed cleanly.")
    return 0

if __name__ == "__main__":
    sys.exit(main())