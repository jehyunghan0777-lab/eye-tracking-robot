from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WorkspaceBounds:
    minimum: np.ndarray
    maximum: np.ndarray


DEFAULT_WORKSPACE_BOUNDS = WorkspaceBounds(
    minimum=np.asarray(
        [0.08, -0.28, -0.05],
        dtype=np.float64,
    ),
    maximum=np.asarray(
        [0.45, 0.28, 0.35],
        dtype=np.float64,
    ),
)


def prepare_approach_point(
    object_point_base: np.ndarray,
    approach_height_meters: float,
    bounds: WorkspaceBounds = DEFAULT_WORKSPACE_BOUNDS,
) -> np.ndarray:
    """Raise a localized object point and enforce the robot workspace."""
    point = np.asarray(
        object_point_base,
        dtype=np.float64,
    )

    if point.shape != (3,):
        raise ValueError(
            "Robot target must contain exactly x, y, and z."
        )

    if not np.all(np.isfinite(point)):
        raise ValueError("Robot target contains a non-finite value.")

    if (
        not np.isfinite(approach_height_meters)
        or approach_height_meters < 0.0
    ):
        raise ValueError(
            "Approach height must be a finite, non-negative value."
        )

    command_point = point.copy()
    command_point[2] += approach_height_meters

    if np.any(command_point < bounds.minimum) or np.any(
        command_point > bounds.maximum
    ):
        raise ValueError(
            "Target is outside the guarded workspace: "
            f"x={command_point[0]:.3f}, "
            f"y={command_point[1]:.3f}, "
            f"z={command_point[2]:.3f} m"
        )

    return command_point
