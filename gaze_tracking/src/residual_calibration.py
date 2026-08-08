from __future__ import annotations

from pathlib import Path
import csv
import sys
import time
import ctypes

import cv2
import mediapipe as mp
import numpy as np

from calibration_test import (
    LEFT_EYE_CORNER_INDEX,
    LEFT_IRIS_CENTER_INDEX,
    RIGHT_EYE_CORNER_INDEX,
    RIGHT_IRIS_CENTER_INDEX,
    calculate_eye_position,
    open_camera,
)

from live_gaze_test import(
    load_regression_model,
    predict_gaze,
)

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]

FACE_LANDMARKER_MODEL_PATH = (
    PROJECT_DIRECTORY
    / "models"
    / "face_landmarker.task"
)

REGRESSION_MODEL_PATH = (
    PROJECT_DIRECTORY
    / "data"
    / "calibration"
    / "gaze_regression_model.json"
)

OUTPUT_PATH = (
    PROJECT_DIRECTORY
    / "data"
    / "calibration"
    / "residual_calibration_samples.csv"
)

SAMPLES_PER_TARGET = 30

WINDOW_NAME = "Residual Calibration"

TARGET_POINTS = [
    (0.1, 0.1),
    (0.5, 0.1),
    (0.9, 0.1),

    (0.1, 0.5),
    (0.5, 0.5),
    (0.9, 0.5),

    (0.1, 0.9),
    (0.5, 0.9),
    (0.9, 0.9),
]

def save_residual_data(
    calibration_results: list[dict],
) -> None:
    """Save second-stage residual calibration data."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "target_x",
        "target_y",
        "predicted_x",
        "predicted_y",
        "residual_x",
        "residual_y",
        "predicted_x_std",
        "predicted_y_std",
        "sample_count",
    ]

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            calibration_results
        )

def main() -> int:
    """Collect residual errors from the baseline gaze model."""

    if not FACE_LANDMARKER_MODEL_PATH.exists():
        print(
            "ERROR: Face Landmarker model was not found at:\n"
            f"{FACE_LANDMARKER_MODEL_PATH}",
            file=sys.stderr,
        )
        return 1

    if not REGRESSION_MODEL_PATH.exists():
        print(
            "ERROR: Regression model was not found at:\n"
            f"{REGRESSION_MODEL_PATH}",
            file=sys.stderr,
        )
        return 1

    regression_model = load_regression_model(
        REGRESSION_MODEL_PATH
    )

    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(
            FACE_LANDMARKER_MODEL_PATH
        )
    )

    options = (
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=(
                mp.tasks.vision.RunningMode.VIDEO
            ),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    camera = open_camera()

    if not camera.isOpened():
        print(
            "ERROR: Could not open webcam.",
            file=sys.stderr
        )
        return 1

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )

    screen_width = (
        ctypes.windll.user32.GetSystemMetrics(0)
    )

    screen_height = (
        ctypes.windll.user32.GetSystemMetrics(1)
    )

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    target_index = 0

    collecting = False
    target_completed = False

    predicted_x_samples = []
    predicted_y_samples = []

    calibration_results = []

    start_time = time.perf_counter()

    with (
        mp.tasks.vision.FaceLandmarker
        .create_from_options(options)
    ) as landmarker:

        while True:

            success, frame = camera.read()

            if not success:
                print(
                    "ERROR: Could not read webcam frame.",
                    file=sys.stderr,
                )
                break

            frame = cv2.flip(
                frame,
                1
            )

            frame_height, frame_width = (
                frame.shape[:2]
            )

            rgb_frame = cv2.cvtColor(
                frame, 
                cv2.COLOR_BGR2RGB,
            )

            mediapipe_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame, 
            )

            timestamp_ms = int(
                (
                    time.perf_counter()
                    - start_time
                ) * 1000
            ) 

            result = (
                landmarker.detect_for_video(
                    mediapipe_image,
                    timestamp_ms,
                )
            )

            canvas = np.full(
                (
                    screen_height,
                    screen_width,
                    3,
                ),
                255,
                dtype=np.uint8,
            )

            target_x, target_y = (
                TARGET_POINTS[target_index]
            )

            target_pixel_x = int(
                target_x * screen_width
            )

            target_pixel_y = int(
                target_y * screen_height
            )

            if collecting:
                target_color = (
                    0,
                    0,
                    255,
                )
            else:
                target_color = (
                    0, 
                    255,
                    0,
                )

            cv2.circle(
                canvas,
                (
                    target_pixel_x,
                    target_pixel_y,
                ),
                18,
                target_color,
                -1,
            )

            cv2.putText(
                canvas,
                (
                    f"Target "
                    f"{target_index + 1}"
                    f"/{len(TARGET_POINTS)}"
                ),
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

            if collecting:
                status_text = (
                    "Keep looking at the target "
                    f"({len(predicted_x_samples)})"
                    f"/{SAMPLES_PER_TARGET}"
                )

            elif target_completed:
                status_text = (
                    "Target complete. "
                    "Press SPACE for next target."
                )

            else:
                status_text = (
                    "Look at the target and "
                    "press SPACE to start."
                )

            cv2.putText(
                canvas,
                status_text,
                (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

            if (
                collecting
                and result.face_landmarks
            ):
                face_landmarks = (
                    result.face_landmarks[0]
                )

                right_eye_position = (
                    calculate_eye_position(
                        face_landmarks,
                        RIGHT_IRIS_CENTER_INDEX,
                        RIGHT_EYE_CORNER_INDEX,
                        frame_width,
                        frame_height,
                    )
                )

                left_eye_position = (
                    calculate_eye_position(
                        face_landmarks,
                        LEFT_IRIS_CENTER_INDEX,
                        LEFT_EYE_CORNER_INDEX,
                        frame_width,
                        frame_height
                    )
                )

                if (
                    right_eye_position is not None
                    and left_eye_position is not None
                ):
                    (
                        right_horizontal,
                        right_vertical,
                    ) = right_eye_position

                    (
                        left_horizontal,
                        left_vertical,
                    ) = left_eye_position

                    eye_horizontal = (
                        right_horizontal
                        + left_horizontal
                    ) / 2.0

                    eye_vertical = (
                        right_vertical
                        + left_vertical
                    ) / 2.0

                    (
                        predicted_x,
                        predicted_y,
                    ) = predict_gaze(
                        eye_horizontal,
                        eye_vertical,
                        regression_model,
                    )

                    predicted_x_samples.append(
                        predicted_x
                    )

                    predicted_y_samples.append(
                        predicted_y
                    )

                    if (
                        len(predicted_x_samples)
                        >= SAMPLES_PER_TARGET
                    ):

                        mean_predicted_x = float(
                            np.mean(
                                predicted_x_samples
                            )
                        )

                        mean_predicted_y = float(
                            np.mean(
                                predicted_y_samples
                            )
                        )

                        predicted_x_std = float(
                            np.std(
                                predicted_x_samples
                            )
                        )

                        predicted_y_std = float(
                            np.std(
                                predicted_y_samples
                            )
                        )

                        residual_x = (
                            target_x 
                            - mean_predicted_x
                        )

                        residual_y = (
                            target_y
                            - mean_predicted_y
                        )

                        calibration_results.append(
                            {
                                "target_x": target_x,
                                "target_y": target_y,
                                "predicted_x": (
                                    mean_predicted_x
                                ),
                                "predicted_y": (
                                    mean_predicted_y
                                ),
                                "residual_x": (
                                    residual_x
                                ),
                                "residual_y": (
                                    residual_y
                                ),
                                "predicted_x_std": (
                                    predicted_x_std
                                ),
                                "predicted_y_std": (
                                    predicted_y_std
                                ),
                                "sample_count": (
                                    SAMPLES_PER_TARGET
                                ),
                            }
                        )

                        print(
                            f"Completed target "
                            f"{target_index + 1}/"
                            f"{len(TARGET_POINTS)}: "
                            f"target="
                            f"({target_x:.3f}, "
                            f"{target_y:.3f}), "
                            f"prediction="
                            f"({mean_predicted_x:.3f}, "
                            f"{mean_predicted_y:.3f}), "
                            f"residual="
                            f"({residual_x:.3f}, "
                            f"{residual_y:.3f}), "
                            f"std="
                            f"({predicted_x_std:.3f}, "
                            f"{predicted_y_std:.3f})"
                        )

                        collecting = False 
                        target_completed = True

            cv2.imshow(
                WINDOW_NAME,
                canvas,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                ord("q"),
                27,
            ):
                print(
                    "Residual calibration cancelled."
                )
                break

            if key == ord(" "):
                if target_completed:

                    if (
                        target_index
                        == len(TARGET_POINTS) - 1
                    ):

                        save_residual_data(
                            calibration_results
                        )

                        print(
                            "\nResidual calibration "
                            "complete."
                        )

                        print(
                            "Saved residual data to:"
                        )

                        print(
                            OUTPUT_PATH
                        )

                        break
                    
                    target_index += 1

                    predicted_x_samples = []
                    predicted_y_samples = []

                    target_completed = False

                elif not collecting:

                    predicted_x_samples = []
                    predicted_y_samples = []

                    collecting = True

                    print (
                        f"Started target "
                        f"{target_index + 1}/"
                        f"{len(TARGET_POINTS)}"
                    )

    camera.release()

    cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())


