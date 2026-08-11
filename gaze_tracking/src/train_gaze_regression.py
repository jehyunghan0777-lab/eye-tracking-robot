from __future__ import annotations

from pathlib import Path
import csv
import json
import sys

import numpy as np

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]

CALIBRATION_DATA_PATH = (
    PROJECT_DIRECTORY
    / "data"
    / "calibration"
    / "calibration_samples.csv"
)

MODEL_OUTPUT_PATH = (
    PROJECT_DIRECTORY
    / "data"
    / "calibration"
    / "gaze_regression_model.json"
)

def  load_calibration_data (
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load eye features and target positions from the calibraiton CSV."""

    eye_features = []
    target_x_values = []
    target_y_values = []

    required_columns = {
        "eye_horizontal",
        "eye_vertical",
        "target_x",
        "target_y"
    }

    with csv_path.open (
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise ValueError(
                "Calibration CSV does not contain a header."
            )

        missing_columns = (
            required_columns - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "Calibration CSV is missing columns: "
                f"{sorted(missing_columns)}"
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            # Ignore completly empty CSV rows.
            if not any(row.values()):
                continue
            
            try:
                eye_horizontal = float(
                    row["eye_horizontal"]
                )

                eye_vertical = float(
                    row["eye_vertical"]
                )

                target_x = float(
                    row["target_x"]
                )

                target_y = float(
                    row["target_y"]
                )

            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid numeric data on CSV row "
                    f"{row_number}."
                ) from error

            eye_features.append(
                [
                    eye_horizontal,
                    eye_vertical,
                ]
            )

            target_x_values.append(target_x)
            target_y_values.append(target_y)

    if len(eye_features) < 3:
        raise ValueError(
            "At least three calibration samples are required."
        )

    return (
        np.asarray(eye_features, dtype=float),
        np.asarray(target_x_values, dtype=float),
        np.asarray(target_y_values, dtype=float),
    )

def fit_linear_model(
    eye_features: np.ndarray,
    target_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one multiple linear-regression model."""

    sample_count = eye_features.shape[0]

    intercept_column = np.ones(
        (sample_count, 1),
        dtype=float,
    )

    design_matrix = np.hstack(
        (
        intercept_column,
        eye_features,
        )
    )

    coefficients, _, rank, _ = np.linalg.lstsq(
        design_matrix,
        target_values,
        rcond=None,
    )

    if rank < design_matrix.shape[1]:
        raise ValueError(
            "Calibration data cannot uniquely determine "
            "all regression coefficients."
        )

    predictions = design_matrix @ coefficients

    return coefficients, predictions

def calculate_metrics (
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> tuple[float, float]:
    """Calculate mean absolute error and root mean sqaured error."""

    errors = true_values - predicted_values

    mean_absolute_error = float(
        np.mean(np.abs(errors))
    )

    root_mean_squared_error = float(
        np.sqrt(np.mean(errors ** 2))
    )

    return (
        mean_absolute_error,
        root_mean_squared_error
    )

def main() -> int:
    """Train and save the gaze regression models."""

    if not CALIBRATION_DATA_PATH.exists():
        print(
            "ERROR: Calibration data was not found at:\n"
            f"{CALIBRATION_DATA_PATH}",
            file=sys.stderr,
        )
        return 1
    
    try:
        (
            eye_features,
            target_x_values,
            target_y_values,
        ) = load_calibration_data(
            CALIBRATION_DATA_PATH
        )

        x_coefficients, predicted_x_values = (
            fit_linear_model(
                eye_features,
                target_x_values,
            )
        )

        y_coefficients, predicted_y_values = (
            fit_linear_model(
                eye_features,
                target_y_values,
            )
        )
        
    except ValueError as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        return 1

    x_mae, x_rmse = calculate_metrics(
        target_x_values,
        predicted_x_values,
    )

    y_mae, y_rmse = calculate_metrics(
        target_y_values,
        predicted_y_values,
    )

    model_data ={
        "model_type": "multiple_linear_regression",
        "feature_order": [
            "eye_horizontal",
            "eye_vertical",
        ],
        "sample_count": int(
            eye_features.shape[0]
        ),
        "screen_x_model": {
            "intercept": float(
                x_coefficients[0]
            ),
            "eye_horizontal_coefficient": float(
                x_coefficients[1]
            ),
            "eye_vertical_coefficient": float(
                x_coefficients[2]
            ),
        },
        "screen_y_model": {
            "intercept": float(
                y_coefficients[0]
            ),
            "eye_horizontal_coefficient": float(
                y_coefficients[1]
            ),
            "eye_vertical_coefficient": float(
                y_coefficients[2]
            ),
        },
        "training_metrics": {
            "screen_x_mae": x_mae,
            "screen_x_rmse": x_rmse,
            "screen_y_mae": y_mae,
            "screen_y_rmse": y_rmse,
        },
    }

    MODEL_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MODEL_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as model_file:
        json.dump(
            model_data,
            model_file,
            indent=4
        )

    print(
        f"Loaded {eye_features.shape[0]} "
        f"calibration samples."
    )

    print("\nScreen-x regression equation:")

    print(
        f"x = {x_coefficients[0]:.6f} "
        f"+ ({x_coefficients[1]:.6f} * H) "
        f"+ ({x_coefficients[2]:.6f} * V)"
    )

    print("\nScreen-y regression equation:")

    print(
        f"y = {y_coefficients[0]:.6f} "
        f"+ ({y_coefficients[1]:.6f} * H) "
        f"+ ({y_coefficients[2]:.6f} * V)"
    )

    print(
        f"Screen x | MAE={x_mae:.4f}, "
        f"RMSE={x_rmse:.4f}"
    )

    print(
        f"Screen y | MAE={y_mae:.4f}, "
        f"RMSE={y_rmse:.4f}"
    )

    print("\nTraining Predictions:")

    for sample_index in range(
        eye_features.shape[0]
    ):
        print(
            f"Sample {sample_index + 1}: "
            f"true=("
            f"{target_x_values[sample_index]:.3f}, "
            f"{target_y_values[sample_index]:.3f}"
            f") "
            f"predicted=("
            f"{predicted_x_values[sample_index]:.3f}, "
            f"{predicted_y_values[sample_index]:.3f}"
            f")"
        )

        print(
            "\nSaved regression model to:\n"
            f"{MODEL_OUTPUT_PATH}"
        )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())



