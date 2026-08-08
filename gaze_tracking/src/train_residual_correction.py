from __future__ import annotations

from pathlib import Path
import csv
import json
import sys

import numpy as np

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]

RESIDUAL_DATA_PATH = (
    PROJECT_DIRECTORY
    / "data"
    / "calibration"
    / "residual_calibration_samples.csv"
)

OUTPUT_MODEL_PATH = (
    PROJECT_DIRECTORY
    / "data"
    / "calibration"
    / "residual_correction_model.json"
)

REQUIRED_COLUMNS = {
    "target_x",
    "target_y",
    "predicted_x",
    "predicted_y",
    "residual_x",
    "residual_y",
}

def load_residual_data(
    csv_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load residual calibration samples."""

    baseline_predictions = []
    targets = []
    residuals = []

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise ValueError(
                "Residual calibration CSV has no header."
            )
        
        missing_columns = (
            REQUIRED_COLUMNS 
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "Residual calibration CSV is missing "
                f"columns: {sorted(missing_columns)}"
            )

        for row in reader:
            
            baseline_predictions.append(
                [
                    float(row["predicted_x"]),
                    float(row["predicted_y"]),
                ]
            )

            targets.append(
                [
                    float(row["target_x"]),
                    float(row["target_y"]),
                ]
            )

            residuals.append(
                [
                    float(row["residual_x"]),
                    float(row["residual_y"]),
                ]
            )

    baseline_predictions = np.array(
        baseline_predictions,
        dtype=float,
    )

    targets = np.array(
        targets,
        dtype=float,
    )

    residuals = np.array(
        residuals,
        dtype=float,
    )

    if len(baseline_predictions) < 7:
        raise ValueError(
            "At least 6 residual calibration samples are required."
        )

    expected_residuals = (
        targets - baseline_predictions
    )

    if not np.allclose(
        residuals,
        expected_residuals,
        atol=1e-6,
    ):
        raise ValueError(
            "Residual values do not match "
            "target - prediction."
        )

    return (
        baseline_predictions,
        targets,
        residuals,
    )

def build_affine_design_matrix(
    baseline_predictions: np.ndarray,
) -> np.ndarray:
    """Create features for the horizontal correction model."""

    baseline_x = baseline_predictions[:,0]
    baseline_y = baseline_predictions[:,1]

    return np.column_stack(
        (
            np.ones(
                len(baseline_predictions),
                dtype=float,
            ),
            baseline_x,
            baseline_y,
        )
    )

def build_quadratic_design_matrix(
    baseline_predictions: np.ndarray,
) -> np.ndarray:
    """Create features for the vertical correction model."""

    baseline_x = baseline_predictions[:,0]
    baseline_y = baseline_predictions[:,1]

    return np.column_stack(
        (
            np.ones(
                len(baseline_predictions),
                dtype=float,
            ),
            baseline_x,
            baseline_y,
            baseline_x ** 2,
            baseline_x * baseline_y,
            baseline_y ** 2,
        )
    )

def fit_least_squares_model(
    design_matrix: np.ndarray,
    target_values: np.ndarray,
) -> np.ndarray:
    """Fit coefficients using least squares."""

    coefficients, _, rank, _ = np.linalg.lstsq(
        design_matrix,
        target_values,
        rcond=None,
    )

    if rank < design_matrix.shape[1]:
        raise ValueError(
            "Design matrix is rank deficient. "
            "The model conet be fitted reliably"
        )
    
    return coefficients

def calculate_metrics(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> tuple[float, float]:
    """Calculate MAE and RMSE."""

    errors = (
        predicted_values - true_values
    )

    mae = float(
        np.mean(
            np.abs(errors)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )

    return mae, rmse

def leave_one_out_metrics(
    baseline_predictions: np.ndarray,
    targets: np.ndarray,
    residual_values: np.ndarray,
    axis: int,
    design_matrix_builder,
) -> tuple[float, float]:
    """Evaluate correction using leave-one-out validation."""

    corrected_values = []
    true_values = []

    sample_count = len(
        baseline_predictions
    )

    for held_out_index in range(
        sample_count
    ):

        training_mask = (
            np.arange(sample_count)
            != held_out_index
        )

        training_predictions = (
            baseline_predictions[
                training_mask
            ]
        )

        training_residuals = (
            residual_values[
                training_mask
            ]
        )

        training_design_matrix = (
            design_matrix_builder(
                training_predictions
            )
        )

        coefficients = (
            fit_least_squares_model(
                training_design_matrix,
                training_residuals,
            )
        )

        test_prediction = (
            baseline_predictions[
                held_out_index:
                held_out_index + 1
            ]
        )

        test_design_matrix = (
            design_matrix_builder(
                test_prediction
            )
        )

        predicted_residual = (
            test_design_matrix
            @ coefficients
        )[0]

        corrected_value = (
            baseline_predictions[
                held_out_index,
                axis,
            ]
            + predicted_residual
        )

        corrected_values.append(
            corrected_value
        )

        true_values.append(
            targets[
                held_out_index,
                axis,
            ]
        )

    corrected_values = np.array(
        corrected_values,
        dtype=float,
    )

    true_values = np.array(
        true_values,
        dtype=float,
    )

    return calculate_metrics(
        true_values,
        corrected_values,
    )

def save_model(
    model_data: dict,
) -> None:
    """Save the residual correction model."""

    OUTPUT_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_MODEL_PATH.open(
        "w",
        encoding="utf-8",
    ) as model_file:

        json.dump(
            model_data,
            model_file,
            indent=4,
        )

def main() -> int:
    """Train the second-stage gaze correction model."""

    if not RESIDUAL_DATA_PATH.exists():
        print(
            "ERROR: Residual calibration data "
            "was not found at:\n"
            f"{RESIDUAL_DATA_PATH}",
            file=sys.stderr,
        )
        return 1

    try:
        (
            baseline_predictions,
            targets,
            residuals,
        ) = load_residual_data(
            RESIDUAL_DATA_PATH
        )

    except (
        OSError,
        ValueError,
    ) as error:

        print(
            f"Error: {error}",
            file=sys.stderr,
        )
        return 1

    sample_count = len(
        baseline_predictions
    )

    print(
        f"Loaded {sample_count} "
        "residual calibraiton samples."
    )

    x_design_matrix = (
        build_affine_design_matrix(
            baseline_predictions
        )
    )

    y_design_matrix = (
        build_quadratic_design_matrix(
            baseline_predictions
        )
    )

    try:
        x_coefficients = (
            fit_least_squares_model(
                x_design_matrix,
                residuals[:, 0],
            )
        )

        y_coefficients = (
            fit_least_squares_model(
                y_design_matrix,
                residuals[:, 1],
            )
        )
    
    except ValueError as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        return 1
    
    predicted_x_residuals = (
        x_design_matrix
        @ x_coefficients
    )

    predicted_y_residuals = (
        y_design_matrix
        @ y_coefficients
    )

    corrected_predictions = (
        baseline_predictions.copy()
    )

    corrected_predictions[:, 0] += (
        predicted_x_residuals
    )

    corrected_predictions[:, 1] += (
        predicted_y_residuals
    )

    baseline_x_mae, baseline_x_rmse = (
        calculate_metrics(
            targets[:, 0],
            baseline_predictions[:, 0],
        )
    )

    baseline_y_mae, baseline_y_rmse = (
        calculate_metrics(
            targets[:, 1],
            baseline_predictions[:, 1],
        )
    )

    corrected_x_mae, corrected_x_rmse = (
        calculate_metrics(
            targets[:, 0],
            corrected_predictions[:, 0],
        )
    )

    corrected_y_mae, corrected_y_rmse = (
        calculate_metrics(
            targets[:, 1],
            corrected_predictions[:, 1],
        )
    )

    loo_x_mae, loo_x_rmse = (
        leave_one_out_metrics(
            baseline_predictions,
            targets,
            residuals[:, 0],
            axis=0,
            design_matrix_builder=(
                build_affine_design_matrix
            ),
        )
    )

    loo_y_mae, loo_y_rmse = (
        leave_one_out_metrics(
            baseline_predictions,
            targets,
            residuals[:, 1],
            axis=1,
            design_matrix_builder=(
                build_quadratic_design_matrix
            ),
        )
    )

    print(
        "\nHorizontal residual model:"
    )

    print(
        "delta_x = "
        f"{x_coefficients[0]:.6f} "
        f"+ ({x_coefficients[1]:.6f} * x) "
        f"+ ({x_coefficients[2]:.6f} * y)"
    )

    print(
        "\nVertical residual model:"
    )

    print(
        "delta_y = "
        f"{y_coefficients[0]:.6f} "
        f"+ ({y_coefficients[1]:.6f} * x) "
        f"+ ({y_coefficients[2]:.6f} * y) "
        f"+ ({y_coefficients[3]:.6f} * x^2) "
        f"+ ({y_coefficients[4]:.6f} * x*y) "
        f"+ ({y_coefficients[5]:.6f} * y^2)"
    )

    print(
        "\nBaseline errors:"
    )

    print(
        f"X | MAE={baseline_x_mae:.4f}, "
        f"RMSE={baseline_x_rmse:.4f}"
    )

    print(
        f"Y | MAE={baseline_y_mae:.4f}, "
        f"RMSE={baseline_y_rmse:.4f}"
    )

    print(
        "\nCorrected training errors:"
    )

    print(
        f"X | MAE={corrected_x_mae:.4f}, "
        f"RMSE={corrected_x_rmse:.4f}"
    )

    print(
        f"Y | MAE={corrected_y_mae:.4f}, "
        f"RMSE={corrected_y_rmse:.4f}"
    )

    print(
        "\nLeave-one-out errors:"
    )

    print(
        f"X | MAE={loo_x_mae:.4f}, "
        f"RMSE={loo_x_rmse:.4f}"
    )

    print(
        f"Y | MAE={loo_y_mae:.4f}, "
        f"RMSE={loo_y_rmse:.4f}"
    )

    model_data = {
        "model_type": (
            "hybrid_residual_correction"
        ),
        "sample_count": sample_count,
        "input_order": [
            "baseline_x",
            "baseline_y",
        ],
        "x_correction_model": {
            "model_type": "affine",
            "intercept": float(
                x_coefficients[0]
            ),
            "baseline_x_coefficient": float(
                x_coefficients[1]
            ),
            "baseline_y_coefficient": float(
                x_coefficients[2]
            ),
        },
        "y_correction_model": {
            "model_type": "quadratic_2d",
            "intercept": float(
                y_coefficients[0]
            ),
            "baseline_x_coefficient": float(
                y_coefficients[1]
            ),
            "baseline_y_coefficient": float(
                y_coefficients[2]
            ),
            "baseline_x_squared_coefficient": float(
                y_coefficients[3]
            ),
            "baseline_xy_coefficient": float(
                y_coefficients[4]
            ),
            "baseline_y_squared_coefficient": float(
                y_coefficients[5]
            ),
        },
        "training_metrics": {
            "baseline_x_mae": (
                baseline_x_mae
            ),
            "baseline_y_mae": (
                baseline_y_mae
            ),
            "corrected_x_mae": (
                corrected_x_mae
            ),
            "corrected_y_mae": (
                corrected_y_mae
            ),
            "leave_one_out_x_mae": (
                loo_x_mae
            ),
            "leave_one_out_y_mae": (
                loo_y_mae
            ),
        },
        "calibration_input_bounds": {
            "baseline_x_min": float(
                np.min(
                    baseline_predictions[:, 0]
                )
            ),
            "baseline_x_max": float(
                np.max(
                    baseline_predictions[:, 0]
                )
            ),
            "baseline_y_min": float(
                np.min(
                    baseline_predictions[:, 1]
                )
            ),
            "baseline_y_max": float(
                np.max(
                    baseline_predictions[:, 1]
                )
            ),
        },
    }

    save_model(
        model_data
    )

    print(
        "\nSaved residual correction model to:"
    )

    print(
        OUTPUT_MODEL_PATH
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

