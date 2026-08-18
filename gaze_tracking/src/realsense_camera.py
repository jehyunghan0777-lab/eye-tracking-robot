from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs


@dataclass(frozen=True)
class RealSenseDeviceInfo:
    name: str
    serial_number: str
    firmware_version: str
    usb_type: str


@dataclass(frozen=True)
class RealSenseFrame:
    color_bgr: np.ndarray
    depth_meters: np.ndarray
    depth_intrinsics: Any
    timestamp_ms: float


class RealSenseCameraError(RuntimeError):
    pass


class RealSenseCamera:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        frames_per_second: int = 30,
        serial_number: str | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(
                "Frame dimensions must be positive."
            )

        if frames_per_second <= 0:
            raise ValueError(
                "Frame rate must be positive."
            )

        self.width = width
        self.height = height
        self.frames_per_second = (
            frames_per_second
        )

        self.serial_number = serial_number

        self.pipeline = rs.pipeline()
        self.configuration = rs.config()

        self.configuration.enable_stream(
            rs.stream.depth,
            width,
            height,
            rs.format.z16,
            frames_per_second,
        )

        self.configuration.enable_stream(
            rs.stream.color,
            width,
            height,
            rs.format.bgr8,
            frames_per_second,
        )

        if serial_number is not None:
            self.configuration.enable_device(
                serial_number
            )

        self.aligner = rs.align(
            rs.stream.color
        )

        self.pipeline_profile = None
        self.depth_scale = None
        self.device_info = None
        self.started = False

    @staticmethod
    def _read_device_information(
        device,
    ) -> RealSenseDeviceInfo:
        def get_information(
            information_type,
            fallback: str = "unknown",
        ) -> str:
            if device.supports(
                information_type
            ):
                return device.get_info(
                    information_type
                )

            return fallback

        return RealSenseDeviceInfo(
            name=get_information(
                rs.camera_info.name
            ),
            serial_number=get_information(
                rs.camera_info.serial_number
            ),
            firmware_version=get_information(
                rs.camera_info.firmware_version
            ),
            usb_type=get_information(
                rs.camera_info.usb_type_descriptor
            ),
        )

    def start(
        self,
        warmup_frames: int = 30,
    ) -> RealSenseDeviceInfo:
        if self.started:
            raise RealSenseCameraError(
                "The camera is already running."
            )

        context = rs.context()
        connected_devices = (
            context.query_devices()
        )

        if len(connected_devices) == 0:
            raise RealSenseCameraError(
                "No RealSense camera was detected."
            )

        try:
            self.pipeline_profile = (
                self.pipeline.start(
                    self.configuration
                )
            )

            self.started = True

            device = (
                self.pipeline_profile
                .get_device()
            )

            self.device_info = (
                self._read_device_information(
                    device
                )
            )

            depth_sensor = (
                device.first_depth_sensor()
            )

            self.depth_scale = (
                depth_sensor.get_depth_scale()
            )

            for _ in range(warmup_frames):
                self.pipeline.wait_for_frames(
                    5000
                )

            return self.device_info

        except RuntimeError as error:
            if self.started:
                self.stop()

            raise RealSenseCameraError(
                f"Could not start the camera: {error}"
            ) from error

    def get_frame(
        self,
        timeout_ms: int = 5000,
    ) -> RealSenseFrame:
        if (
            not self.started
            or self.depth_scale is None
        ):
            raise RealSenseCameraError(
                "The camera has not been started."
            )

        try:
            frames = (
                self.pipeline.wait_for_frames(
                    timeout_ms
                )
            )

            aligned_frames = (
                self.aligner.process(frames)
            )

            aligned_depth_frame = (
                aligned_frames.get_depth_frame()
            )

            color_frame = (
                aligned_frames.get_color_frame()
            )

            if (
                not aligned_depth_frame
                or not color_frame
            ):
                raise RealSenseCameraError(
                    "An aligned color/depth "
                    "frame was unavailable."
                )

            color_bgr = np.asanyarray(
                color_frame.get_data()
            )

            raw_depth = np.asanyarray(
                aligned_depth_frame.get_data()
            )

            depth_meters = (
                raw_depth.astype(np.float32)
                * self.depth_scale
            )

            depth_intrinsics = (
                aligned_depth_frame.profile
                .as_video_stream_profile()
                .intrinsics
            )

            return RealSenseFrame(
                color_bgr=color_bgr,
                depth_meters=depth_meters,
                depth_intrinsics=(
                    depth_intrinsics
                ),
                timestamp_ms=(
                    aligned_frames.get_timestamp()
                ),
            )

        except RuntimeError as error:
            raise RealSenseCameraError(
                f"Could not obtain a frame: {error}"
            ) from error

    @staticmethod
    def median_depth_around_pixel(
        frame: RealSenseFrame,
        pixel_x: int,
        pixel_y: int,
        radius: int = 3,
        minimum_depth: float = 0.05,
        maximum_depth: float = 3.0,
    ) -> float | None:
        if radius < 0:
            raise ValueError(
                "Radius cannot be negative."
            )

        frame_height, frame_width = (
            frame.depth_meters.shape
        )

        if not (
            0 <= pixel_x < frame_width
            and 0 <= pixel_y < frame_height
        ):
            return None

        x_min = max(
            0,
            pixel_x - radius,
        )

        x_max = min(
            frame_width,
            pixel_x + radius + 1,
        )

        y_min = max(
            0,
            pixel_y - radius,
        )

        y_max = min(
            frame_height,
            pixel_y + radius + 1,
        )

        depth_patch = frame.depth_meters[
            y_min:y_max,
            x_min:x_max,
        ]

        valid_depths = depth_patch[
            np.isfinite(depth_patch)
            & (depth_patch >= minimum_depth)
            & (depth_patch <= maximum_depth)
        ]

        if valid_depths.size == 0:
            return None

        return float(
            np.median(valid_depths)
        )

    @staticmethod
    def deproject_pixel(
        frame: RealSenseFrame,
        pixel_x: int,
        pixel_y: int,
        depth_meters: float,
    ) -> tuple[float, float, float]:
        if not np.isfinite(depth_meters):
            raise ValueError(
                "Depth must be finite."
            )

        if depth_meters <= 0.0:
            raise ValueError(
                "Depth must be positive."
            )

        camera_point = (
            rs.rs2_deproject_pixel_to_point(
                frame.depth_intrinsics,
                [
                    float(pixel_x),
                    float(pixel_y),
                ],
                float(depth_meters),
            )
        )

        return (
            float(camera_point[0]),
            float(camera_point[1]),
            float(camera_point[2]),
        )

    def stop(self) -> None:
        if self.started:
            self.pipeline.stop()
            self.started = False

    def __enter__(
        self,
    ) -> RealSenseCamera:
        self.start()
        return self

    def __exit__(
        self,
        exception_type,
        exception_value,
        traceback,
    ) -> None:
        self.stop()


def colorize_depth(
    depth_meters: np.ndarray,
    maximum_display_depth: float,
) -> np.ndarray:
    clipped_depth = np.clip(
        depth_meters,
        0.0,
        maximum_display_depth,
    )

    normalized_depth = (
        clipped_depth
        / maximum_display_depth
        * 255.0
    ).astype(np.uint8)

    return cv2.applyColorMap(
        normalized_depth,
        cv2.COLORMAP_TURBO,
    )


def run_preview(
    serial_number: str | None,
    width: int,
    height: int,
    frames_per_second: int,
    maximum_display_depth: float,
) -> int:
    camera = RealSenseCamera(
        width=width,
        height=height,
        frames_per_second=(
            frames_per_second
        ),
        serial_number=serial_number,
    )

    try:
        device_information = camera.start()

        print("RealSense camera started")
        print(
            f"Name: "
            f"{device_information.name}"
        )
        print(
            f"Serial number: "
            f"{device_information.serial_number}"
        )
        print(
            f"Firmware: "
            f"{device_information.firmware_version}"
        )
        print(
            f"USB type: "
            f"{device_information.usb_type}"
        )
        print(
            f"Depth scale: "
            f"{camera.depth_scale}"
        )
        print(
            "Press Q or Escape to stop."
        )

        while True:
            frame = camera.get_frame()

            display_color = (
                frame.color_bgr.copy()
            )

            center_x = width // 2
            center_y = height // 2

            center_depth = (
                camera.median_depth_around_pixel(
                    frame,
                    center_x,
                    center_y,
                    radius=3,
                )
            )

            cv2.drawMarker(
                display_color,
                (center_x, center_y),
                (0, 255, 255),
                cv2.MARKER_CROSS,
                24,
                2,
            )

            if center_depth is None:
                depth_text = (
                    "Center depth: unavailable"
                )
            else:
                camera_point = (
                    camera.deproject_pixel(
                        frame,
                        center_x,
                        center_y,
                        center_depth,
                    )
                )

                depth_text = (
                    f"Depth: {center_depth:.3f} m  "
                    f"XYZ: "
                    f"({camera_point[0]:.3f}, "
                    f"{camera_point[1]:.3f}, "
                    f"{camera_point[2]:.3f}) m"
                )

            cv2.putText(
                display_color,
                depth_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            depth_color = colorize_depth(
                frame.depth_meters,
                maximum_display_depth,
            )

            combined_display = np.hstack(
                (
                    display_color,
                    depth_color,
                )
            )

            cv2.imshow(
                "RealSense Color and Aligned Depth",
                combined_display,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    except RealSenseCameraError as error:
        print(f"ERROR: {error}")
        return 1

    finally:
        camera.stop()
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Preview aligned RealSense "
            "color and depth frames."
        )
    )

    parser.add_argument(
        "--serial-number",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--maximum-display-depth",
        type=float,
        default=2.0,
    )

    arguments = parser.parse_args()

    return run_preview(
        serial_number=arguments.serial_number,
        width=arguments.width,
        height=arguments.height,
        frames_per_second=arguments.fps,
        maximum_display_depth=(
            arguments.maximum_display_depth
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())