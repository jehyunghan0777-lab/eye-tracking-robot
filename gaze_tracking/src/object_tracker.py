from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class DetectedObject:
    track_id: int | None
    label: str
    confidence: float
    bounding_box: tuple[int, int, int, int]
    mask: np.ndarray | None

    def contains_pixel(
        self,
        pixel_x: int,
        pixel_y: int,
    ) -> bool:
        x_min, y_min, x_max, y_max = self.bounding_box

        return (
            x_min <= pixel_x <= x_max
            and y_min <= pixel_y <= y_max
        )

    @property
    def center(self) -> tuple[int, int]:
        x_min, y_min, x_max, y_max = self.bounding_box

        return (
            (x_min + x_max) // 2,
            (y_min + y_max) // 2,
        )


class ObjectTracker:
    def __init__(
        self,
        model_name: str = "yolo26n-seg.pt",
        confidence_threshold: float = 0.35,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "Confidence threshold must be between 0 and 1."
            )

        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold

    def track(
        self,
        frame: np.ndarray,
    ) -> list[DetectedObject]:
        results = self.model.track(
            source=frame,
            persist=True,
            conf=self.confidence_threshold,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        frame_height, frame_width = frame.shape[:2]

        mask_data = None

        if result.masks is not None:
            mask_data = result.masks.data.cpu().numpy()

        detected_objects = []

        for detection_index, box in enumerate(result.boxes):
            coordinates = box.xyxy[0].cpu().numpy()

            x_min = max(0, int(coordinates[0]))
            y_min = max(0, int(coordinates[1]))

            x_max = min(
                frame_width - 1,
                int(coordinates[2]),
            )

            y_max = min(
                frame_height - 1,
                int(coordinates[3]),
            )

            class_id = int(box.cls.item())
            label = str(result.names[class_id])
            confidence = float(box.conf.item())

            track_id = None

            if box.id is not None:
                track_id = int(box.id.item())

            mask = None

            if (
                mask_data is not None
                and detection_index < len(mask_data)
            ):
                resized_mask = cv2.resize(
                    mask_data[detection_index],
                    (frame_width, frame_height),
                    interpolation=cv2.INTER_NEAREST,
                )

                mask = resized_mask > 0.5

            detected_objects.append(
                DetectedObject(
                    track_id=track_id,
                    label=label,
                    confidence=confidence,
                    bounding_box=(
                        x_min,
                        y_min,
                        x_max,
                        y_max,
                    ),
                    mask=mask,
                )
            )

        return detected_objects