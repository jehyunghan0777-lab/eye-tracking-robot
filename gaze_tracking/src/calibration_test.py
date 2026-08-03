from __future__ import annotations

from pathlib import Path
import sys
import time
import csv

import cv2
import mediapipe as mp
import numpy as np

WINDOW_NAME = "Eye-Tracking Robot - Gaze Calibration"

# __file__is this Python file.
# parents[0] is src/
# parents[1] is gaze_tracking/
MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "face_landmarker.task"
)

CALIBRATION_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "calibration"
    / "calibration_samples.csv"
)

# Target position is normalized from 0 to 1.
CALIBRATION_TARGETS = [
    (0.1, 0.1),
    (0.5, 0.1),
    (0.9, 0.1),

    (0.1,0.5),
    (0.5, 0.5),
    (0.9, 0.5),

    (0.1, 0.9),
    (0.5, 0.9),
    (0.9, 0.9),
]

TARGET_OUTER_RADIUS = 20
TARGET_INNER_RADIUS = 6

# Number of frames to sample per target.
SAMPLES_PER_TARGET = 30

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

def draw_calibration_target(
    canvas,
    normalized_position,
    target_color,
) -> None:
    """Draw a calibration target at a normalized screen position."""

    canvas_height, canvas_width = canvas.shape[:2]

    normalized_x, normalized_y = normalized_position

    target_x = int(normalized_x * canvas_width)
    target_y = int(normalized_y * canvas_height)

    cv2.circle(
        canvas,
        (target_x, target_y),
        TARGET_OUTER_RADIUS,
        target_color,  
        -1,  
        cv2.LINE_AA,
    )

    cv2.circle(
        canvas,
        (target_x, target_y),
        TARGET_INNER_RADIUS,
        (255, 255, 255),  # White center.
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
    

def save_calibration_samples(
    calibration_samples,
) -> None: 
    """Save collected calibration samples to CSV file."""

    CALIBRATION_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CALIBRATION_OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "eye_horizontal",
                "eye_vertical",
                "target_x",
                "target_y",
            ]
        )

        writer.writerows(calibration_samples)

    print(
        f"Saved calibration data to:\n"
        f"{CALIBRATION_OUTPUT_PATH}"
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

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

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

    current_target_index = 0
    calibration_samples = []

    target_eye_samples = []
    calibration_state = "ready"

    try:
        with FaceLandmarker.create_from_options(options) as landmarker:
            print("Face Landmarker intialized.")
            print("Press Q or ESC to quit.")

            while True:
                current_eye_position = None
                status_text = "No valid eye position"

                # Determine which target is active
                current_target = CALIBRATION_TARGETS[
                    current_target_index
                ]

                # Read and process the camera frame. 
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


                        # Calculate average horizontal and vertical positon of iris.               
                        average_horizontal = (
                            right_horizontal + left_horizontal
                        ) / 2.0

                        average_vertical = (
                            right_vertical + left_vertical
                        ) / 2.0

                        current_eye_position = (
                            average_horizontal,
                            average_vertical,
                        )

                        status_text = (
                            f"Eye Position | "
                            f"H: {average_horizontal:.3f} "
                            f"V: {average_vertical:.3f}"
                        )
                        
                        # Collect one valid measurement per video frame.
                        if calibration_state == "collecting":
                            target_eye_samples.append(
                                current_eye_position
                            )

                        # Finish this target after SAMPLE_PER_TARGET valid frames.
                        if (
                            len(
                            target_eye_samples) 
                            >= SAMPLES_PER_TARGET
                        ):

                            average_target_horizontal = sum(
                                sample[0] for sample in target_eye_samples
                            ) / len(target_eye_samples)

                            average_target_vertical = sum(
                                sample[1] for sample in target_eye_samples
                            ) / len(target_eye_samples)

                            target_x, target_y = current_target

                            calibration_samples.append(
                                [
                                    average_target_horizontal,
                                    average_target_vertical,
                                    target_x,
                                    target_y,
                                ]
                            )

                            print(
                                f"Completed target "
                                f"{current_target_index + 1}/"
                                f"{len(CALIBRATION_TARGETS)}: "
                                f"H={average_target_horizontal:.3f}, "
                                f"V={average_target_vertical:.3f}, "
                                f"target=({target_x:.1f}, "
                                f"{target_y:.1f}), "
                                f"samples={len(target_eye_samples)}"
                            )

                            target_eye_samples.clear()    
                            calibration_state = "complete"
                    
                    else: 
                        status_text = "Eye landmarks unavailable"
                        
                else:
                    status_text = "No face detected"

                # Create black calibration screen.
                display_frame = np.zeros_like(frame)

                # Red while collecting; green otherwise. 
                if calibration_state == "collecting":
                    target_color = (0, 0, 255)
                else:  
                    target_color = (0, 255, 0)

                draw_calibration_target(
                    display_frame,
                    current_target,
                    target_color,
                )

                # Select instructions for the current state.
                if calibration_state == "ready":
                    instruction_text = (
                        f"Target {current_target_index + 1}/"
                        f"{len(CALIBRATION_TARGETS)} "
                        f"| Look at target and press Space"
                    )

                elif calibration_state == "collecting":
                    instruction_text = (
                        f"Collecting: "
                        f"{len(target_eye_samples)}/"
                        f"{SAMPLES_PER_TARGET} "
                        f"| Keep looking at the target"
                    )

                else:
                    is_last_target = (
                        current_target_index
                        == len(CALIBRATION_TARGETS) - 1
                    )

                    if is_last_target:
                            instruction_text = (
                                "Calibration complete | "
                                "Press Space to save and exit"
                            )
                    else:
                        instruction_text = (
                            "Target complete |"
                            "Press Space for the next target"
                        )

                cv2.putText(
                    display_frame,
                    instruction_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    display_frame, 
                    status_text,
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    display_frame, 
                    "Q/Esc: exit",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(
                    WINDOW_NAME,
                    display_frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:  # ESC
                    print("Exiting...")
                    break

                if key == ord(" "):
                    if calibration_state == "ready":
                        if current_eye_position is None:
                            print("No valid eye position. Sample not recorded")
                        else:
                            target_eye_samples.clear()
                            calibration_state = "collecting"

                            print(
                                f"Started target "
                                f"{current_target_index + 1}/"
                                f"{len(CALIBRATION_TARGETS)}"
                            )
                    
                    elif calibration_state == "collecting":
                        print(
                            "Calibration is already collecting samples."
                        )
                    
                    elif calibration_state == "complete":
                        is_last_target = (
                            current_target_index
                            == len(CALIBRATION_TARGETS) - 1
                        )

                        if is_last_target:
                            save_calibration_samples(
                                calibration_samples
                            )
                            break
                        
                        current_target_index += 1
                        calibration_state = "ready"

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("Calibration test closed cleanly.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
