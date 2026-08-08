from __future__ import annotations

from pathlib import Path
import time
import json
import sys
import cv2
import mediapipe as mp

from calibration_test import (
    LEFT_EYE_CORNER_INDEX,
    LEFT_IRIS_CENTER_INDEX,
    RIGHT_EYE_CORNER_INDEX,
    RIGHT_IRIS_CENTER_INDEX,
    calculate_eye_position,
    open_camera
)


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_DIRECTORY
    / "data"
    / "calibration"
    / "gaze_regression_model.json"
)

FACE_LANDMARKER_MODEL_PATH = (
    PROJECT_DIRECTORY
    / "models"
    / "face_landmarker.task"
)

def load_regression_model(
    model_path: Path,
) -> dict:
    """Load the trained gaze regression model."""

    with model_path.open(
        "r",
        encoding="utf-8",
    ) as model_file:
        model_data = json.load(model_file)

    return model_data

def predict_gaze(
    eye_horizontal: float,
    eye_vertical: float,
    model_data: dict,
) -> tuple[float, float]:
    """Predict normalized screen coordinates from eye features."""

    x_model = model_data["screen_x_model"]
    y_model = model_data["screen_y_model"]

    predicted_x = (
        x_model["intercept"]
        + x_model["eye_horizontal_coefficient"] * eye_horizontal
        + x_model["eye_vertical_coefficient"] * eye_vertical
    )

    predicted_y = (
        y_model["intercept"]
        + y_model["eye_horizontal_coefficient"] * eye_horizontal
        + y_model["eye_vertical_coefficient"] * eye_vertical
    )

    return predicted_x, predicted_y

def main() -> int:
    """Run live gaze prediction from the webcam."""

    if not MODEL_PATH.exists():
        print(
            "ERROR: Regression model was not found at:\n"
            f"{MODEL_PATH}",
            file=sys.stderr,
        )
        return 1

    if not FACE_LANDMARKER_MODEL_PATH.exists():
        print(
            "ERROR: Face Landmarker model was not found at:\n"
            f"{FACE_LANDMARKER_MODEL_PATH}",
            file=sys.stderr,
        )
        return 1

    model_data = load_regression_model(MODEL_PATH)

    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(FACE_LANDMARKER_MODEL_PATH)
    )

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
    )

    camera = open_camera()

    if camera is None:
        return 1

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    start_time = time.perf_counter()

    WINDOW_NAME = "Live Gaze Test"

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        1280,
        720,
    )

    with mp.tasks.vision.FaceLandmarker.create_from_options(
        options
    ) as landmarker:
        
        while True:
            success, frame = camera.read()

            if not success:
                print(
                    "ERROR: Could not read webcame frame.",
                    file=sys.stderr,
                )
                break

            frame = cv2.flip(frame, 1)

            frame_height, frame_width = frame.shape[:2]

            rgb_frame = cv2.cvtColor(
                frame, 
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            timestamp_ms = int(
                (time.perf_counter() - start_time) * 1000
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0]

                right_horizontal, right_vertical = (
                    calculate_eye_position(
                        face_landmarks,
                        RIGHT_IRIS_CENTER_INDEX,
                        RIGHT_EYE_CORNER_INDEX,
                        frame_width,
                        frame_height,
                    )
                )

                left_horizontal, left_vertical = (
                    calculate_eye_position(
                        face_landmarks,
                        LEFT_IRIS_CENTER_INDEX,
                        LEFT_EYE_CORNER_INDEX,
                        frame_width,
                        frame_height,
                    )
                )

                eye_horizontal = (
                    right_horizontal + left_horizontal
                ) / 2

                eye_vertical = (
                    right_vertical + left_vertical
                ) / 2

                predicted_x, predicted_y = predict_gaze(
                    eye_horizontal,
                    eye_vertical,
                    model_data,
                )

                clamped_x = max(0.0, min(1.0, predicted_x))
                clamped_y = max(0.0, min(1.0, predicted_y))

                gaze_pixel_x = int(clamped_x * frame_width)
                gaze_pixel_y = int(clamped_y * frame_height)

                cv2.circle(
                    frame,
                    (gaze_pixel_x, gaze_pixel_y),
                    15,
                    (0, 0, 255),
                    -1,
                )

                cv2.putText(
                    frame,
                    (
                        f"H={eye_horizontal:.3f} "
                        f"V={eye_vertical:.3f}"
                    ),
                    (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame, 
                    (
                        f"Gaze x={predicted_x:.3f} "
                        f"y={predicted_y:.3f}"
                    ),
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(
                WINDOW_NAME,
                frame, 
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    camera.release()
    cv2.destroyAllWindows()

    return 0




if __name__ == "__main__":
    raise SystemExit(main())