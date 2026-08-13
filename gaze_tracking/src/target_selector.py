from __future__ import annotations

from dataclasses import dataclass
import time

from object_tracker import DetectedObject

@dataclass
class SelectionResult:
    gaze_pixel: tuple[int, int]
    candidate: DetectedObject | None
    selected: DetectedObject | None
    dwell_progress: float

class TargetSelector:
    def __init__(
        self,
        dwell_seconds: float = 0.75,
        margin_pixels: int = 30,
    ) -> None:
        if dwell_seconds <= 0:
            raise ValueError(
                "Dwell duration must be greater than zero."
            )

        if margin_pixels < 0:
            raise ValueError(
                "Selection margin cannot be negative."
            )

        self.dwell_seconds = dwell_seconds
        self.margin_pixels = margin_pixels

        self.candidate_key: tuple | None = None
        self.candidate_start_time: float | None = None

    def update(
        self,
        gaze_x: float,
        gaze_y: float,
        frame_width: int,
        frame_height: int,
        detected_objects: list[DetectedObject]
    ) -> SelectionResult:
        gaze_x = max(0.0, min(1.0, gaze_x))
        gaze_y = max(0.0, min(1.0, gaze_y))

        pixel_x = min(
            frame_width - 1,
            int(gaze_x * frame_width),
        )

        pixel_y = min(
            frame_height - 1,
            int(gaze_y * frame_height),
        )

        candidate = self._find_candidate(
            pixel_x,
            pixel_y,
            detected_objects
        )

        if candidate is None:
            self._reset_candidate()

            return SelectionResult(
                gaze_pixel=(pixel_x, pixel_y),
                candidate=None,
                selected=None,
                dwell_progress=0.0,
            )

        current_time = time.perf_counter()
        candidate_key = self._object_key(candidate)

        if candidate_key != self.candidate_key:
            self.candidate_key = candidate_key
            self.candidate_start_time = current_time
            dwell_progress = 0.0
        else:
            elapsed_time = (
                current_time
                - self.candidate_start_time
            )

            dwell_progress = min(
                1.0,
                elapsed_time / self.dwell_seconds,
            )

        selected = None

        if dwell_progress >= 1.0:
            selected = candidate

        return SelectionResult(
            gaze_pixel=(pixel_x, pixel_y),
            candidate=candidate,
            selected=selected,
            dwell_progress=dwell_progress
        )

    def _find_candidate(
        self,
        pixel_x: int,
        pixel_y: int,
        detected_objects: list[DetectedObject],
    ) -> DetectedObject | None:
        mask_matches = []

        for detected_object in detected_objects:
            if(
                detected_object.mask is not None
                and detected_object.mask[
                    pixel_y,
                    pixel_x
                ]
            ):
                mask_matches.append(detected_object)

        if mask_matches:
            return min(
                mask_matches,
                key=lambda detected_object: (
                    self._distance_squared(
                        detected_object,
                        pixel_x,
                        pixel_y,
                    )
                ),
            )

        box_matches = []

        for detected_object in detected_objects:
            x_min, y_min, x_max, y_max = (
                detected_object.bounding_box
            )

            if (
                x_min - self.margin_pixels
                <= pixel_x
                <= x_max + self.margin_pixels
                and y_min - self.margin_pixels
                <= pixel_y
                <= y_max + self.margin_pixels
            ):
                box_matches.append(detected_object)

        if not box_matches:
            return None

        return min(
            box_matches,
            key=lambda detected_object: (
                self._distance_squared(
                    detected_object,
                    pixel_x,
                    pixel_y,
                )
            ),
        )

    @staticmethod
    def _distance_squared(
        detected_object: DetectedObject,
        pixel_x: int,
        pixel_y: int,
    ) -> int:
        center_x, center_y = detected_object.center

        return (
            (pixel_x - center_x) ** 2
            + (pixel_y - center_y) ** 2
        )

    @staticmethod
    def _object_key(
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

    def _reset_candidate(self) -> None:
        self.candidate_key = None
        self.candidate_start_time = None