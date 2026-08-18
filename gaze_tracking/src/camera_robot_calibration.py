from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from realsense_camera import (
    RealSenseCamera,
    RealSenseCameraError,
)


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_PATH = (
    PROJECT_DIRECTORY
    / "config"
    / "camera_to_robot_transform.json"
)

WINDOW_NAME = "D435 to Robot Calibration"


@dataclass
class MouseState:
    clicked_pixel: tuple[int, int] | None = None


@dataclass(frozen=True)
class CalibrationResult:
    rotation_matrix: np.ndarray
    translation_meters: np.ndarray
    rmse_meters: float
    maximum_error_meters: float
    errors_meters: np.ndarray


def mouse_callback(
    event: int,
    pixel_x: int,
    pixel_y: int,
    flags: int,
    mouse_state: MouseState,
) -> None:
    del flags

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_state.clicked_pixel = (
            pixel_x,
            pixel_y,
        )


def read_base_point() -> np.ndarray | None:
    print()
    print(
        "Enter the matching robot base_link point "
        "as: x y z"
    )
    print("Example: 0.350 -0.100 0.150")
    print("Press Enter without values to cancel.")

    raw_value = input("base_link XYZ (meters): ").strip()

    if not raw_value:
        return None

    fields = raw_value.replace(",", " ").split()

    if len(fields) != 3:
        print("ERROR: Enter exactly three coordinates.")
        return None

    try:
        point = np.asarray(
            [float(field) for field in fields],
            dtype=np.float64,
        )
    except ValueError:
        print("ERROR: Coordinates must be numbers.")
        return None

    if not np.all(np.isfinite(point)):
        print("ERROR: Coordinates must be finite.")
        return None

    return point


def solve_rigid_transform(
    camera_points: np.ndarray,
    base_points: np.ndarray,
) -> CalibrationResult:
    if camera_points.shape != base_points.shape:
        raise ValueError(
            "Camera and robot point arrays must match."
        )

    if (
        camera_points.ndim != 2
        or camera_points.shape[1] != 3
    ):
        raise ValueError(
            "Calibration points must have shape N x 3."
        )

    if camera_points.shape[0] < 4:
        raise ValueError(
            "At least four point pairs are required."
        )

    camera_centroid = np.mean(
        camera_points,
        axis=0,
    )

    base_centroid = np.mean(
        base_points,
        axis=0,
    )

    centered_camera = (
        camera_points - camera_centroid
    )

    centered_base = (
        base_points - base_centroid
    )

    if np.linalg.matrix_rank(centered_camera) < 2:
        raise ValueError(
            "Calibration points are nearly collinear. "
            "Spread them across the workspace."
        )

    covariance = (
        centered_camera.T
        @ centered_base
    )

    left_vectors, _, right_vectors_transposed = (
        np.linalg.svd(covariance)
    )

    rotation_matrix = (
        right_vectors_transposed.T
        @ left_vectors.T
    )

    if np.linalg.det(rotation_matrix) < 0.0:
        right_vectors_transposed[-1, :] *= -1.0

        rotation_matrix = (
            right_vectors_transposed.T
            @ left_vectors.T
        )

    translation_meters = (
        base_centroid
        - rotation_matrix @ camera_centroid
    )

    predicted_base_points = (
        (
            rotation_matrix
            @ camera_points.T
        ).T
        + translation_meters
    )

    errors_meters = np.linalg.norm(
        predicted_base_points - base_points,
        axis=1,
    )

    rmse_meters = float(
        np.sqrt(
            np.mean(errors_meters ** 2)
        )
    )

    maximum_error_meters = float(
        np.max(errors_meters)
    )

    return CalibrationResult(
        rotation_matrix=rotation_matrix,
        translation_meters=translation_meters,
        rmse_meters=rmse_meters,
        maximum_error_meters=(
            maximum_error_meters
        ),
        errors_meters=errors_meters,
    )


def transform_camera_point(
    camera_point: np.ndarray,
    calibration: CalibrationResult,
) -> np.ndarray:
    return (
        calibration.rotation_matrix
        @ camera_point
        + calibration.translation_meters
    )


def save_calibration(
    output_path: Path,
    calibration: CalibrationResult,
    camera_points: np.ndarray,
    base_points: np.ndarray,
    camera_serial_number: str,
) -> None:
    payload = {
        "source_frame": "camera_color_optical_frame",
        "target_frame": "base_link",
        "camera_serial_number": camera_serial_number,
        "point_count": int(camera_points.shape[0]),
        "rotation_matrix": (
            calibration.rotation_matrix.tolist()
        ),
        "translation_meters": (
            calibration.translation_meters.tolist()
        ),
        "rmse_meters": calibration.rmse_meters,
        "maximum_error_meters": (
            calibration.maximum_error_meters
        ),
        "errors_meters": (
            calibration.errors_meters.tolist()
        ),
        "camera_points_meters": (
            camera_points.tolist()
        ),
        "base_points_meters": (
            base_points.tolist()
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as output_file:
        json.dump(
            payload,
            output_file,
            indent=2,
        )

        output_file.write("\n")


def load_calibration(
    input_path: Path = DEFAULT_OUTPUT_PATH,
) -> CalibrationResult:
    with input_path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        payload = json.load(input_file)

    rotation_matrix = np.asarray(
        payload["rotation_matrix"],
        dtype=np.float64,
    )

    translation_meters = np.asarray(
        payload["translation_meters"],
        dtype=np.float64,
    )

    errors_meters = np.asarray(
        payload.get("errors_meters", []),
        dtype=np.float64,
    )

    if rotation_matrix.shape != (3, 3):
        raise ValueError(
            "Calibration rotation matrix must be 3 x 3."
        )

    if translation_meters.shape != (3,):
        raise ValueError(
            "Calibration translation must contain 3 values."
        )

    if (
        not np.all(np.isfinite(rotation_matrix))
        or not np.all(np.isfinite(translation_meters))
    ):
        raise ValueError(
            "Calibration contains non-finite values."
        )

    determinant = float(
        np.linalg.det(rotation_matrix)
    )

    if not np.isclose(
        determinant,
        1.0,
        atol=1e-3,
    ):
        raise ValueError(
            "Calibration rotation matrix is invalid."
        )

    return CalibrationResult(
        rotation_matrix=rotation_matrix,
        translation_meters=translation_meters,
        rmse_meters=float(
            payload.get("rmse_meters", 0.0)
        ),
        maximum_error_meters=float(
            payload.get(
                "maximum_error_meters",
                0.0,
            )
        ),
        errors_meters=errors_meters,
    )


def draw_calibration_interface(
    color_image: np.ndarray,
    selected_pixel: tuple[int, int] | None,
    selected_camera_point: np.ndarray | None,
    recorded_point_count: int,
) -> np.ndarray:
    display_image = color_image.copy()

    instructions = [
        "Click robot tip | A: add pair | U: undo",
        "S: solve and save | Q/Esc: quit",
        f"Recorded pairs: {recorded_point_count}",
    ]

    for line_index, instruction in enumerate(
        instructions
    ):
        cv2.putText(
            display_image,
            instruction,
            (20, 35 + line_index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

    if selected_pixel is not None:
        cv2.drawMarker(
            display_image,
            selected_pixel,
            (0, 0, 255),
            cv2.MARKER_CROSS,
            30,
            3,
        )

    if (
        selected_pixel is not None
        and selected_camera_point is not None
    ):
        camera_x, camera_y, camera_z = (
            selected_camera_point
        )

        coordinate_text = (
            f"Camera XYZ: {camera_x:.3f}, "
            f"{camera_y:.3f}, {camera_z:.3f} m"
        )

        cv2.putText(
            display_image,
            coordinate_text,
            (
                max(10, selected_pixel[0] - 180),
                max(120, selected_pixel[1] - 20),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )

    return display_image


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the fixed transform from a "
            "RealSense camera to robot base_link."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    parser.add_argument(
        "--serial-number",
        default=None,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
    )

    arguments = parser.parse_args()

    camera = RealSenseCamera(
        width=arguments.width,
        height=arguments.height,
        frames_per_second=arguments.fps,
        serial_number=arguments.serial_number,
    )

    try:
        device_information = camera.start()
    except RealSenseCameraError as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        return 1

    print("RealSense camera started")
    print(f"Name: {device_information.name}")
    print(
        f"Serial: "
        f"{device_information.serial_number}"
    )
    print()
    print(
        "Use at least six points spread across the "
        "robot workspace."
    )
    print(
        "Include points at different heights when possible."
    )

    mouse_state = MouseState()

    camera_points: list[np.ndarray] = []
    base_points: list[np.ndarray] = []

    selected_pixel: tuple[int, int] | None = None
    selected_camera_point: np.ndarray | None = None

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback,
        mouse_state,
    )

    try:
        while True:
            frame = camera.get_frame()

            if mouse_state.clicked_pixel is not None:
                selected_pixel = (
                    mouse_state.clicked_pixel
                )

                mouse_state.clicked_pixel = None

                depth_meters = (
                    RealSenseCamera
                    .median_depth_around_pixel(
                        frame,
                        selected_pixel[0],
                        selected_pixel[1],
                        radius=5,
                    )
                )

                if depth_meters is None:
                    selected_camera_point = None

                    print(
                        "No valid depth at the selected pixel."
                    )
                else:
                    selected_camera_point = np.asarray(
                        RealSenseCamera.deproject_pixel(
                            frame,
                            selected_pixel[0],
                            selected_pixel[1],
                            depth_meters,
                        ),
                        dtype=np.float64,
                    )

                    print(
                        "Selected camera point: "
                        f"{selected_camera_point.tolist()}"
                    )

            display_image = draw_calibration_interface(
                frame.color_bgr,
                selected_pixel,
                selected_camera_point,
                len(camera_points),
            )

            cv2.imshow(
                WINDOW_NAME,
                display_image,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("a"):
                if selected_camera_point is None:
                    print(
                        "Select a valid camera pixel first."
                    )
                    continue

                base_point = read_base_point()

                if base_point is None:
                    print("Point was not recorded.")
                    continue

                camera_points.append(
                    selected_camera_point.copy()
                )

                base_points.append(base_point)

                print(
                    f"Recorded pair {len(camera_points)}"
                )
                print(
                    f"  Camera: "
                    f"{camera_points[-1].tolist()}"
                )
                print(
                    f"  base_link: "
                    f"{base_points[-1].tolist()}"
                )

                selected_pixel = None
                selected_camera_point = None

            elif key == ord("u"):
                if camera_points:
                    camera_points.pop()
                    base_points.pop()

                    print("Removed the most recent pair.")
                else:
                    print("There are no points to remove.")

            elif key == ord("s"):
                if len(camera_points) < 4:
                    print(
                        "At least four pairs are required. "
                        f"Currently recorded: {len(camera_points)}"
                    )
                    continue

                camera_array = np.vstack(
                    camera_points
                )

                base_array = np.vstack(
                    base_points
                )

                try:
                    calibration = solve_rigid_transform(
                        camera_array,
                        base_array,
                    )
                except ValueError as error:
                    print(f"ERROR: {error}")
                    continue

                save_calibration(
                    arguments.output,
                    calibration,
                    camera_array,
                    base_array,
                    device_information.serial_number,
                )

                print()
                print("Calibration saved")
                print(f"File: {arguments.output}")
                print(
                    f"RMSE: "
                    f"{calibration.rmse_meters * 1000.0:.1f} mm"
                )
                print(
                    "Maximum error: "
                    f"{calibration.maximum_error_meters * 1000.0:.1f} mm"
                )
                print("Rotation matrix:")
                print(calibration.rotation_matrix)
                print("Translation (meters):")
                print(calibration.translation_meters)

            elif key in {
                ord("q"),
                27,
            }:
                break

    except RealSenseCameraError as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        return 1

    except KeyboardInterrupt:
        pass

    finally:
        camera.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())