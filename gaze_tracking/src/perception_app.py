from __future__ import annotations

import sys
import time

import cv2
import mediapipe as mp

from calibration_test import (
    LEFT_EYE_CORNER_INDEX,
    LEFT_IRIS_CENTER_INDEX,
    RIGHT_EYE_CORNER_INDEX,
    RIGHT_IRIS_CENTER_INDEX,
    calculate_eye_position,
    open_camera,
)

from live_gaze_test import (
    FACE_LANDMARKER_MODEL_PATH,
    MODEL_PATH,
    RESIDUAL_MODEL_PATH,
    apply_residual_correction,
    load_regression_model,
    load_residual_model,
    predict_gaze,
)

from object_tracker import (
    DetectedObject,
    ObjectTracker,
)

from realsense_camera import (
    RealSenseCamera,
    RealSenseCameraError,
    RealSenseFrame,
)

from rgbd_target_localizer import (
    LocalizedTarget,
    RgbdTargetLocalizer,
    draw_localized_target,
)

from target_selector import (
    SelectionResult,
    TargetSelector,
)


WINDOW_NAME = "Gaze Object Selection"

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

D435_FRAMES_PER_SECOND = 30


def validate_required_files() -> bool:
    required_files = {
        "gaze regression model": MODEL_PATH,
        "residual correction model": RESIDUAL_MODEL_PATH,
        "MediaPipe face model": FACE_LANDMARKER_MODEL_PATH,
    }

    for description, file_path in required_files.items():
        if not file_path.exists():
            print(
                f"ERROR: {description} not found:\n"
                f"{file_path}",
                file=sys.stderr,
            )
            return False

    return True


def estimate_gaze(
    face_frame,
    landmarker,
    timestamp_ms: int,
    regression_model: dict,
    residual_model: dict,
) -> tuple[float, float] | None:
    frame_height, frame_width = face_frame.shape[:2]

    rgb_frame = cv2.cvtColor(
        face_frame,
        cv2.COLOR_BGR2RGB,
    )

    media_pipe_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame,
    )

    result = landmarker.detect_for_video(
        media_pipe_image,
        timestamp_ms,
    )

    if not result.face_landmarks:
        return None

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

    baseline_x, baseline_y = predict_gaze(
        eye_horizontal,
        eye_vertical,
        regression_model,
    )

    corrected_x, corrected_y = (
        apply_residual_correction(
            baseline_x,
            baseline_y,
            residual_model,
        )
    )

    return corrected_x, corrected_y


def object_color(
    detected_object: DetectedObject,
    selection: SelectionResult | None,
) -> tuple[int, int, int]:
    if (
        selection is not None
        and detected_object is selection.selected
    ):
        return (0, 0, 255)

    if (
        selection is not None
        and detected_object is selection.candidate
    ):
        return (0, 255, 255)

    return (0, 255, 0)


def draw_interface(
    scene_frame,
    detected_objects: list[DetectedObject],
    selection: SelectionResult | None,
):
    display_frame = scene_frame.copy()

    for detected_object in detected_objects:
        color = object_color(
            detected_object,
            selection,
        )

        if detected_object.mask is not None:
            overlay = display_frame.copy()
            overlay[detected_object.mask] = color

            display_frame = cv2.addWeighted(
                overlay,
                0.20,
                display_frame,
                0.80,
                0.0,
            )

        x_min, y_min, x_max, y_max = (
            detected_object.bounding_box
        )

        cv2.rectangle(
            display_frame,
            (x_min, y_min),
            (x_max, y_max),
            color,
            3,
        )

        track_text = ""

        if detected_object.track_id is not None:
            track_text = (
                f" ID:{detected_object.track_id}"
            )

        label_text = (
            f"{detected_object.label}"
            f"{track_text} "
            f"{detected_object.confidence:.2f}"
        )

        cv2.putText(
            display_frame,
            label_text,
            (x_min, max(25, y_min - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )

    if selection is None:
        cv2.putText(
            display_frame,
            "Face not detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )

        return display_frame

    gaze_x, gaze_y = selection.gaze_pixel

    cv2.circle(
        display_frame,
        (gaze_x, gaze_y),
        10,
        (255, 255, 255),
        -1,
    )

    cv2.circle(
        display_frame,
        (gaze_x, gaze_y),
        13,
        (0, 0, 0),
        2,
    )

    progress_angle = int(
        360 * selection.dwell_progress
    )

    cv2.ellipse(
        display_frame,
        (gaze_x, gaze_y),
        (24, 24),
        0,
        -90,
        -90 + progress_angle,
        (0, 255, 255),
        4,
    )

    status_text = "Look at an object"
    status_color = (255, 255, 255)

    if selection.candidate is not None:
        status_text = (
            f"Candidate: "
            f"{selection.candidate.label} "
            f"({selection.dwell_progress:.0%})"
        )

        status_color = (0, 255, 255)

    if selection.selected is not None:
        status_text = (
            f"SELECTED: "
            f"{selection.selected.label}"
        )

        status_color = (0, 0, 255)

    cv2.putText(
        display_frame,
        status_text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        2,
    )

    return display_frame


def localize_selected_object(
    localizer: RgbdTargetLocalizer,
    rgbd_frame: RealSenseFrame,
    selected_object: DetectedObject,
) -> LocalizedTarget | None:
    if selected_object.mask is None:
        print(
            "ERROR: The selected object does not "
            "have a segmentation mask.",
            file=sys.stderr,
        )
        return None

    target = localizer.localize(
        rgbd_frame,
        selected_object.mask,
    )

    if target is None:
        print(
            "ERROR: No valid depth was found for "
            f"'{selected_object.label}'.",
            file=sys.stderr,
        )
        return None

    print(
        f"Localized '{selected_object.label}' in "
        f"camera frame: x={target.camera_x:.3f}, "
        f"y={target.camera_y:.3f}, "
        f"z={target.camera_z:.3f} m"
    )

    return target


def selection_key(
    detected_object: DetectedObject,
) -> tuple:
    if detected_object.track_id is not None:
        return (
            "track",
            detected_object.track_id,
        )

    center_x, center_y = detected_object.center

    return (
        "fallback",
        detected_object.label,
        center_x // 80,
        center_y // 80,
    )


def main() -> int:
    if not validate_required_files():
        return 1

    regression_model = load_regression_model(
        MODEL_PATH
    )

    residual_model = load_residual_model(
        RESIDUAL_MODEL_PATH
    )

    print("Loading YOLO segmentation model...")

    object_tracker = ObjectTracker(
        confidence_threshold=0.15,
    )

    target_localizer = RgbdTargetLocalizer()

    target_selector = TargetSelector(
        dwell_seconds=0.75,
        margin_pixels=30,
    )

    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(
            FACE_LANDMARKER_MODEL_PATH
        )
    )

    landmarker_options = (
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=(
                mp.tasks.vision.RunningMode.VIDEO
            ),
            num_faces=1,
        )
    )

    face_camera = open_camera()

    if face_camera is None:
        return 1

    face_camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )

    face_camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )

    scene_camera = RealSenseCamera(
        width=DISPLAY_WIDTH,
        height=DISPLAY_HEIGHT,
        frames_per_second=D435_FRAMES_PER_SECOND,
    )

    try:
        device_information = scene_camera.start()
    except RealSenseCameraError as error:
        face_camera.release()
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        return 1

    print(
        f"RealSense started: "
        f"{device_information.name}, "
        f"serial {device_information.serial_number}"
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

    start_time = time.perf_counter()

    last_selected_key: tuple | None = None
    localized_target: LocalizedTarget | None = None

    try:
        with (
            mp.tasks.vision.FaceLandmarker
            .create_from_options(
                landmarker_options
            )
        ) as landmarker:

            while True:
                rgbd_frame = scene_camera.get_frame()
                scene_frame = rgbd_frame.color_bgr

                detected_objects = object_tracker.track(
                    scene_frame
                )

                success, face_frame = (
                    face_camera.read()
                )

                if not success:
                    print(
                        "ERROR: Could not read webcam frame.",
                        file=sys.stderr,
                    )
                    break

                face_frame = cv2.flip(
                    face_frame,
                    1,
                )

                timestamp_ms = int(
                    (
                        time.perf_counter()
                        - start_time
                    )
                    * 1000
                )

                gaze = estimate_gaze(
                    face_frame,
                    landmarker,
                    timestamp_ms,
                    regression_model,
                    residual_model,
                )

                if gaze is None:
                    target_selector.reset()
                    selection = None
                else:
                    corrected_x, corrected_y = gaze

                    selection = target_selector.update(
                        corrected_x,
                        corrected_y,
                        DISPLAY_WIDTH,
                        DISPLAY_HEIGHT,
                        detected_objects,
                    )

                display_frame = draw_interface(
                    scene_frame,
                    detected_objects,
                    selection,
                )

                if (
                    selection is None
                    or selection.candidate is None
                ):
                    last_selected_key = None
                    localized_target = None
                elif (
                    selection.selected is not None
                ):
                    selected_object = selection.selected
                    selected_key = selection_key(
                        selected_object
                    )

                    if selected_key != last_selected_key:
                        localized_target = (
                            localize_selected_object(
                                target_localizer,
                                rgbd_frame,
                                selected_object,
                            )
                        )

                        last_selected_key = selected_key

                if localized_target is not None:
                    display_frame = draw_localized_target(
                        display_frame,
                        localized_target,
                    )

                cv2.imshow(
                    WINDOW_NAME,
                    display_frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if (
                    key == ord("q")
                    or key == 27
                ):
                    break

    except RealSenseCameraError as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        return 1

    finally:
        scene_camera.stop()
        face_camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())