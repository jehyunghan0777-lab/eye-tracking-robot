from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from realsense_camera import (
    RealSenseCamera,
    RealSenseFrame,
)


@dataclass(frozen=True)
class LocalizedTarget:
    pixel_x: int
    pixel_y: int
    depth_meters: float

    camera_x: float
    camera_y: float
    camera_z: float

    valid_depth_pixels: int


class RgbdTargetLocalizer:
    def __init__(
        self,
        min_depth_meters: float = 0.10,
        max_depth_meters: float = 2.00,
        erosion_kernel_size: int = 9,
        minimum_valid_pixels: int = 20,
    ) -> None:
        self.min_depth_meters = min_depth_meters
        self.max_depth_meters = max_depth_meters
        self.minimum_valid_pixels = minimum_valid_pixels

        if erosion_kernel_size < 1:
            raise ValueError(
                "erosion_kernel_size must be positive"
            )

        if erosion_kernel_size % 2 == 0:
            erosion_kernel_size += 1

        self.erosion_kernel = np.ones(
            (
                erosion_kernel_size,
                erosion_kernel_size,
            ),
            dtype=np.uint8,
        )

    def localize(
        self,
        frame: RealSenseFrame,
        object_mask: np.ndarray,
    ) -> LocalizedTarget | None:
        depth_image = frame.depth_meters

        if object_mask.shape != depth_image.shape:
            raise ValueError(
                "Object mask and depth image must have "
                "the same dimensions"
            )

        binary_mask = (
            object_mask.astype(bool).astype(np.uint8)
        )

        if not np.any(binary_mask):
            return None

        # Remove mask edges because boundary pixels often
        # contain background depth.
        eroded_mask = cv2.erode(
            binary_mask,
            self.erosion_kernel,
            iterations=1,
        )

        if np.count_nonzero(eroded_mask) < (
            self.minimum_valid_pixels
        ):
            eroded_mask = binary_mask

        valid_mask = (
            eroded_mask.astype(bool)
            & np.isfinite(depth_image)
            & (
                depth_image
                >= self.min_depth_meters
            )
            & (
                depth_image
                <= self.max_depth_meters
            )
        )

        valid_count = int(
            np.count_nonzero(valid_mask)
        )

        if valid_count < self.minimum_valid_pixels:
            return None

        valid_depths = depth_image[valid_mask]

        median_depth = float(
            np.median(valid_depths)
        )

        # Reject depth outliers using the median absolute
        # deviation.
        absolute_deviation = np.abs(
            valid_depths - median_depth
        )

        median_deviation = float(
            np.median(absolute_deviation)
        )

        depth_tolerance = max(
            0.02,
            3.0 * 1.4826 * median_deviation,
        )

        robust_mask = (
            valid_mask
            & (
                np.abs(
                    depth_image - median_depth
                )
                <= depth_tolerance
            )
        )

        robust_y, robust_x = np.nonzero(
            robust_mask
        )

        if len(robust_x) < self.minimum_valid_pixels:
            robust_y, robust_x = np.nonzero(
                valid_mask
            )

        pixel_x = int(np.median(robust_x))
        pixel_y = int(np.median(robust_y))

        depth_meters = float(
            np.median(
                depth_image[
                    robust_y,
                    robust_x,
                ]
            )
        )

        camera_x, camera_y, camera_z = (
            RealSenseCamera.deproject_pixel(
                frame,
                pixel_x,
                pixel_y,
                depth_meters,
            )
        )

        return LocalizedTarget(
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            depth_meters=depth_meters,
            camera_x=camera_x,
            camera_y=camera_y,
            camera_z=camera_z,
            valid_depth_pixels=len(robust_x),
        )


def draw_localized_target(
    image,
    target: LocalizedTarget,
):
    output = image.copy()

    cv2.drawMarker(
        output,
        (target.pixel_x, target.pixel_y),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        24,
        3,
    )

    text = (
        f"Camera XYZ: "
        f"{target.camera_x:.3f}, "
        f"{target.camera_y:.3f}, "
        f"{target.camera_z:.3f} m"
    )

    cv2.putText(
        output,
        text,
        (
            max(10, target.pixel_x - 180),
            max(30, target.pixel_y - 20),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
    )

    return output