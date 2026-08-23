from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
)
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEMG_DIRECTORY = Path(__file__).resolve().parents[1]
RAW_DATA_DIRECTORY = SEMG_DIRECTORY / "data" / "raw"
MODEL_DIRECTORY = SEMG_DIRECTORY / "models"


# -------------------------------------------------------------------------
# 60 Hz causal IIR notch filter
#
# Designed for:
#   Sample rate = 1000 Hz
#   Notch frequency = 60 Hz
#   Q = 30
#
# IMPORTANT:
# These coefficients must remain identical to the
# coefficients used by semg_classifier.c on the STM32.
# -------------------------------------------------------------------------

NOTCH_B0 = 0.99375596
NOTCH_B1 = -1.84794186
NOTCH_B2 = 0.99375596

NOTCH_A1 = -1.84794186
NOTCH_A2 = 0.98751193


CLASS_NAMES = [
    "CLOSE",
    "EXTEND",
    "REST",
]


FEATURE_NAMES = [
    "ch1_rms",
    "ch1_mav",
    "ch1_std",
    "ch1_waveform_length",
    "ch1_zero_crossing_rate",
    "ch1_slope_change_rate",
    "ch2_rms",
    "ch2_mav",
    "ch2_std",
    "ch2_waveform_length",
    "ch2_zero_crossing_rate",
    "ch2_slope_change_rate",
]


def find_latest_recording() -> Path:
    csv_files = sorted(
        RAW_DATA_DIRECTORY.glob(
            "semg_training_*.csv"
        ),
        key=lambda path: path.stat().st_mtime,
    )

    if not csv_files:
        raise FileNotFoundError(
            "No training CSV was found in "
            f"{RAW_DATA_DIRECTORY}"
        )

    return csv_files[-1]


def load_segments(
    input_path: Path,
) -> dict[tuple[int, str], np.ndarray]:
    segment_rows = defaultdict(list)

    with input_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as input_file:
        reader = csv.DictReader(input_file)

        required_columns = {
            "trial",
            "label",
            "ch1",
            "ch2",
        }

        if not required_columns.issubset(
            reader.fieldnames or []
        ):
            raise ValueError(
                "The CSV does not contain the "
                "required columns."
            )

        for row in reader:
            trial_number = int(row["trial"])
            label = row["label"].strip().upper()
            channel_1 = float(row["ch1"])
            channel_2 = float(row["ch2"])

            if label not in CLASS_NAMES:
                continue

            segment_rows[
                (trial_number, label)
            ].append(
                (
                    channel_1,
                    channel_2,
                )
            )

    segments = {}

    for segment_key, rows in segment_rows.items():
        segments[segment_key] = np.asarray(
            rows,
            dtype=np.float64,
        )

    return segments


# -------------------------------------------------------------------------
# NEW
#
# Apply the same causal 60 Hz notch filter that runs
# sample-by-sample on the STM32.
#
# Difference equation:
#
# y[n] =
#     b0*x[n]
#   + b1*x[n-1]
#   + b2*x[n-2]
#   - a1*y[n-1]
#   - a2*y[n-2]
#
# We initialize the filter around the current ADC baseline
# instead of zero, matching the STM32 implementation.
# -------------------------------------------------------------------------

def apply_notch_filter(
    signal: np.ndarray,
) -> np.ndarray:
    filtered_signal = np.empty_like(
        signal,
        dtype=np.float64,
    )

    if len(signal) == 0:
        return filtered_signal

    first_sample = float(signal[0])

    x1 = first_sample
    x2 = first_sample

    y1 = first_sample
    y2 = first_sample

    filtered_signal[0] = first_sample

    for index in range(
        1,
        len(signal),
    ):
        input_sample = float(
            signal[index]
        )

        output_sample = (
            NOTCH_B0 * input_sample
            + NOTCH_B1 * x1
            + NOTCH_B2 * x2
            - NOTCH_A1 * y1
            - NOTCH_A2 * y2
        )

        x2 = x1
        x1 = input_sample

        y2 = y1
        y1 = output_sample

        filtered_signal[index] = (
            output_sample
        )

    return filtered_signal


def extract_channel_features(
    signal: np.ndarray,
    crossing_threshold: float,
) -> list[float]:
    centered_signal = signal - np.mean(signal)

    signal_difference = np.diff(
        centered_signal
    )

    rms = np.sqrt(
        np.mean(centered_signal**2)
    )

    mav = np.mean(
        np.abs(centered_signal)
    )

    standard_deviation = np.std(
        centered_signal
    )

    waveform_length = np.mean(
        np.abs(signal_difference)
    )

    crossing_values = (
        centered_signal[:-1]
        * centered_signal[1:]
    )

    zero_crossings = (
        (crossing_values < 0)
        & (
            np.abs(signal_difference)
            >= crossing_threshold
        )
    )

    zero_crossing_rate = np.mean(
        zero_crossings
    )

    first_slopes = signal_difference[:-1]
    second_slopes = signal_difference[1:]

    slope_changes = (
        (first_slopes * second_slopes < 0)
        & (
            np.abs(
                first_slopes
                - second_slopes
            )
            >= crossing_threshold
        )
    )

    slope_change_rate = np.mean(
        slope_changes
    )

    return [
        float(rms),
        float(mav),
        float(standard_deviation),
        float(waveform_length),
        float(zero_crossing_rate),
        float(slope_change_rate),
    ]


def extract_window_features(
    window: np.ndarray,
    crossing_threshold: float,
) -> list[float]:
    channel_1_features = (
        extract_channel_features(
            window[:, 0],
            crossing_threshold,
        )
    )

    channel_2_features = (
        extract_channel_features(
            window[:, 1],
            crossing_threshold,
        )
    )

    return (
        channel_1_features
        + channel_2_features
    )


def create_feature_dataset(
    segments: dict[
        tuple[int, str],
        np.ndarray,
    ],
    window_samples: int,
    step_samples: int,
    crossing_threshold: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    features = []
    labels = []
    trial_groups = []

    for (
        trial_number,
        label,
    ), segment in sorted(segments.items()):

        # -------------------------------------------------------------
        # NEW
        #
        # Filter the COMPLETE trial first.
        #
        # We do NOT reset the notch filter every 200-sample window,
        # because the STM32 filter runs continuously sample-by-sample.
        # -------------------------------------------------------------

        filtered_segment = np.empty_like(
            segment,
            dtype=np.float64,
        )

        filtered_segment[:, 0] = (
            apply_notch_filter(
                segment[:, 0]
            )
        )

        filtered_segment[:, 1] = (
            apply_notch_filter(
                segment[:, 1]
            )
        )

        if len(filtered_segment) < window_samples:
            print(
                "Skipping short segment: "
                f"trial={trial_number}, "
                f"label={label}, "
                f"samples={len(filtered_segment)}"
            )
            continue

        final_start = (
            len(filtered_segment)
            - window_samples
        )

        for start_index in range(
            0,
            final_start + 1,
            step_samples,
        ):
            end_index = (
                start_index
                + window_samples
            )

            # ---------------------------------------------------------
            # CHANGED
            #
            # Feature extraction now uses filtered EMG rather than
            # the original raw ADC samples.
            # ---------------------------------------------------------

            window = filtered_segment[
                start_index:end_index
            ]

            window_features = (
                extract_window_features(
                    window,
                    crossing_threshold,
                )
            )

            features.append(window_features)
            labels.append(label)
            trial_groups.append(trial_number)

    if not features:
        raise ValueError(
            "No feature windows could be created."
        )

    return (
        np.asarray(
            features,
            dtype=np.float64,
        ),
        np.asarray(labels),
        np.asarray(
            trial_groups,
            dtype=np.int32,
        ),
    )


def create_classifier():
    return make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
            priors=np.full(
                len(CLASS_NAMES),
                1.0 / len(CLASS_NAMES),
            ),
        ),
    )


def validate_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    trial_groups: np.ndarray,
    segments: dict[
        tuple[int, str],
        np.ndarray,
    ],
) -> tuple[float, np.ndarray]:
    trial_gestures = {}

    for trial_number, label in segments:
        if label != "REST":
            trial_gestures[
                trial_number
            ] = label

    unique_trials = np.asarray(
        sorted(trial_gestures),
        dtype=np.int32,
    )

    trial_labels = np.asarray(
        [
            trial_gestures[trial]
            for trial in unique_trials
        ]
    )

    gesture_counts = [
        np.count_nonzero(
            trial_labels == gesture
        )
        for gesture in (
            "CLOSE",
            "EXTEND",
        )
    ]

    number_of_folds = min(
        5,
        min(gesture_counts),
    )

    if number_of_folds < 2:
        raise ValueError(
            "At least two CLOSE trials and "
            "two EXTEND trials are required."
        )

    cross_validator = StratifiedKFold(
        n_splits=number_of_folds,
        shuffle=True,
        random_state=42,
    )

    predictions = np.empty(
        labels.shape,
        dtype=object,
    )

    for (
        training_trial_indices,
        validation_trial_indices,
    ) in cross_validator.split(
        unique_trials,
        trial_labels,
    ):
        training_trials = unique_trials[
            training_trial_indices
        ]

        validation_trials = unique_trials[
            validation_trial_indices
        ]

        training_mask = np.isin(
            trial_groups,
            training_trials,
        )

        validation_mask = np.isin(
            trial_groups,
            validation_trials,
        )

        classifier = create_classifier()

        classifier.fit(
            features[training_mask],
            labels[training_mask],
        )

        predictions[validation_mask] = (
            classifier.predict(
                features[validation_mask]
            )
        )

    balanced_accuracy = (
        balanced_accuracy_score(
            labels,
            predictions,
        )
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=CLASS_NAMES,
    )

    print()
    print("Cross-validation results")
    print("------------------------")
    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy:.1%}"
    )
    print()

    print(
        classification_report(
            labels,
            predictions,
            labels=CLASS_NAMES,
            digits=3,
            zero_division=0,
        )
    )

    print("Confusion matrix")
    print(
        "Rows = actual, "
        "columns = predicted"
    )
    print(
        "Class order: "
        + ", ".join(CLASS_NAMES)
    )
    print(matrix)

    return balanced_accuracy, matrix


def write_model_files(
    classifier,
    balanced_accuracy: float,
    confusion: np.ndarray,
    sample_rate: int,
    window_samples: int,
    step_samples: int,
    crossing_threshold: float,
) -> tuple[Path, Path]:
    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    scaler = classifier.named_steps[
        "standardscaler"
    ]

    lda = classifier.named_steps[
        "lineardiscriminantanalysis"
    ]

    combined_weights = (
        lda.coef_
        / scaler.scale_[None, :]
    )

    combined_bias = (
        lda.intercept_
        - combined_weights
        @ scaler.mean_
    )

    json_path = (
        MODEL_DIRECTORY
        / "semg_lda_model.json"
    )

    model_data = {
        "sample_rate_hz": sample_rate,
        "window_samples": window_samples,
        "step_samples": step_samples,
        "crossing_threshold": (
            crossing_threshold
        ),

        # Record the preprocessing used to train this model.
        "notch_filter": {
            "enabled": True,
            "frequency_hz": 60.0,
            "b": [
                NOTCH_B0,
                NOTCH_B1,
                NOTCH_B2,
            ],
            "a": [
                1.0,
                NOTCH_A1,
                NOTCH_A2,
            ],
        },

        "feature_names": FEATURE_NAMES,

        "class_names": (
            lda.classes_.tolist()
        ),

        "weights": (
            combined_weights.tolist()
        ),

        "bias": combined_bias.tolist(),

        "balanced_accuracy": (
            balanced_accuracy
        ),

        "confusion_matrix": (
            confusion.tolist()
        ),
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            model_data,
            output_file,
            indent=2,
        )

    header_path = (
        MODEL_DIRECTORY
        / "semg_lda_model.h"
    )

    with header_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        output_file.write(
            "#ifndef SEMG_LDA_MODEL_H\n"
        )

        output_file.write(
            "#define SEMG_LDA_MODEL_H\n\n"
        )

        output_file.write(
            f"#define SEMG_CLASS_COUNT "
            f"{len(lda.classes_)}\n"
        )

        output_file.write(
            f"#define SEMG_FEATURE_COUNT "
            f"{len(FEATURE_NAMES)}\n"
        )

        output_file.write(
            f"#define SEMG_WINDOW_SAMPLES "
            f"{window_samples}\n\n"
        )

        output_file.write(
            "static const char *const "
            "SEMG_CLASS_NAMES"
            "[SEMG_CLASS_COUNT] = {\n"
        )

        for class_name in lda.classes_:
            output_file.write(
                f'    "{class_name}",\n'
            )

        output_file.write("};\n\n")

        output_file.write(
            "static const float "
            "SEMG_LDA_WEIGHTS"
            "[SEMG_CLASS_COUNT]"
            "[SEMG_FEATURE_COUNT] = {\n"
        )

        for weight_row in combined_weights:
            formatted_weights = ", ".join(
                f"{value:.9g}f"
                for value in weight_row
            )

            output_file.write(
                f"    {{{formatted_weights}}},\n"
            )

        output_file.write("};\n\n")

        formatted_bias = ", ".join(
            f"{value:.9g}f"
            for value in combined_bias
        )

        output_file.write(
            "static const float "
            "SEMG_LDA_BIAS"
            "[SEMG_CLASS_COUNT] = {\n"
        )

        output_file.write(
            f"    {formatted_bias}\n"
        )

        output_file.write("};\n\n")

        output_file.write(
            "static inline int "
            "semg_lda_predict("
            "const float features"
            "[SEMG_FEATURE_COUNT])\n"
        )

        output_file.write("{\n")

        output_file.write(
            "    int best_class = 0;\n"
        )

        output_file.write(
            "    float best_score = "
            "-3.402823466e+38F;\n\n"
        )

        output_file.write(
            "    for (int class_index = 0; "
            "class_index < SEMG_CLASS_COUNT; "
            "class_index++) {\n"
        )

        output_file.write(
            "        float score = "
            "SEMG_LDA_BIAS[class_index];\n\n"
        )

        output_file.write(
            "        for (int feature_index = 0; "
            "feature_index < SEMG_FEATURE_COUNT; "
            "feature_index++) {\n"
        )

        output_file.write(
            "            score += "
            "SEMG_LDA_WEIGHTS"
            "[class_index][feature_index] "
            "* features[feature_index];\n"
        )

        output_file.write("        }\n\n")

        output_file.write(
            "        if (score > best_score) {\n"
        )

        output_file.write(
            "            best_score = score;\n"
        )

        output_file.write(
            "            best_class = class_index;\n"
        )

        output_file.write("        }\n")

        output_file.write("    }\n\n")

        output_file.write(
            "    return best_class;\n"
        )

        output_file.write("}\n\n")

        output_file.write(
            "#endif\n"
        )

    return json_path, header_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train and export the "
            "three-class sEMG classifier."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        help=(
            "Training CSV. If omitted, "
            "the newest recording is used."
        ),
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--window-ms",
        type=float,
        default=200.0,
    )

    parser.add_argument(
        "--step-ms",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--crossing-threshold",
        type=float,
        default=10.0,
    )

    arguments = parser.parse_args()

    input_path = (
        arguments.input
        if arguments.input is not None
        else find_latest_recording()
    )

    window_samples = round(
        arguments.sample_rate
        * arguments.window_ms
        / 1000.0
    )

    step_samples = round(
        arguments.sample_rate
        * arguments.step_ms
        / 1000.0
    )

    print(f"Loading:\n{input_path}")

    segments = load_segments(input_path)

    features, labels, trial_groups = (
        create_feature_dataset(
            segments=segments,
            window_samples=window_samples,
            step_samples=step_samples,
            crossing_threshold=(
                arguments.crossing_threshold
            ),
        )
    )

    print()
    print(
        f"Created {len(features)} "
        "feature windows."
    )

    for class_name in CLASS_NAMES:
        class_windows = np.count_nonzero(
            labels == class_name
        )

        print(
            f"{class_name}: "
            f"{class_windows} windows"
        )

    balanced_accuracy, confusion = (
        validate_classifier(
            features=features,
            labels=labels,
            trial_groups=trial_groups,
            segments=segments,
        )
    )

    final_classifier = create_classifier()

    final_classifier.fit(
        features,
        labels,
    )

    json_path, header_path = (
        write_model_files(
            classifier=final_classifier,

            balanced_accuracy=(
                balanced_accuracy
            ),

            confusion=confusion,

            sample_rate=(
                arguments.sample_rate
            ),

            window_samples=window_samples,

            step_samples=step_samples,

            crossing_threshold=(
                arguments.crossing_threshold
            ),
        )
    )

    print()
    print("Model training completed.")

    print(
        f"JSON model:\n{json_path}"
    )

    print(
        f"STM32 header:\n{header_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())