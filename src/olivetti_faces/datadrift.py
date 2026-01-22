import numpy as np
import pandas as pd
import torch
from evidently.presets import DataDriftPreset
from evidently import Report
import sys
from loguru import logger
import re
from pathlib import Path
import argparse


# Reference: fixed baseline used for training
logger.info("Loading reference and current data for drift monitoring...")
refference_olivetti_images = torch.load("data/processed/refference_images.pt")  # shape: (N, 1, H, W)
current_data_olivetti_images = torch.load("data/processed/train_images.pt") 

#test with drift data
#current_data_olivetti_images = torch.load("data/processed/drift_images.pt") 

# Convert to numpy in shape (N, H, W) so your feature extraction works nicely
refference_olivetti_images = refference_olivetti_images.squeeze(1).numpy()
current_data_olivetti_images = current_data_olivetti_images.squeeze(1).numpy()



def extract_features(images: np.ndarray) -> np.ndarray:
    """
    Extract basic image features from a set of images.
    images: shape (N, H, W), values typically in [0, 1]
    """
    features = []
    for img in images:
        avg_brightness = float(np.mean(img))
        contrast = float(np.std(img))

        # gradient magnitude as a crude "sharpness"
        gx, gy = np.gradient(img)
        sharpness = float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))

        features.append([avg_brightness, contrast, sharpness])
    return np.array(features, dtype=np.float64)

logger.info("Extracting features from images for drift monitoring...")
refference_feature = extract_features(refference_olivetti_images)
current_data_features = extract_features(current_data_olivetti_images)

feature_columns = ["Average Brightness", "Contrast", "Sharpness"]

logger.debug(f"Feature columns: {feature_columns}")
# --- 3) Build dataframes for Evidently ---
reference_df = pd.DataFrame(refference_feature, columns=feature_columns)
current_df = pd.DataFrame(current_data_features, columns=feature_columns)

_DRIFT_SCORE_RE = re.compile(r"Drift score is ([0-9]+(?:\.[0-9]+)?)")
_ACTUAL_VALUE_RE = re.compile(r"Actual value ([0-9]+(?:\.[0-9]+)?)")

def has_data_drift(reference_df, current_df) -> bool:
    evaluation = Report([DataDriftPreset()], include_tests=True).run(
        reference_data=reference_df,
        current_data=current_df,
    )
    d = evaluation.dict()

    tests = d.get("tests") or d.get("test_results")
    if not isinstance(tests, list):
        raise TypeError(f"Unexpected tests type: {type(tests)}")

    drift_detected = False
    logger.info("Drift test summary:")

    for t in tests:
        if not isinstance(t, dict):
            continue

        name = t.get("name", "UNKNOWN TEST")
        status = (t.get("status") or "").upper()
        description = t.get("description", "")
        params = (t.get("metric_config") or {}).get("params", {})

        # Overall drift-share gate
        if "Share of Drifted Columns" in name:
            drift_share_threshold = params.get("drift_share")
            m = _ACTUAL_VALUE_RE.search(description)
            actual_share = float(m.group(1)) if m else None

            logger.info(
                f"Drifted feature share: actual={actual_share}, "
                f"threshold={drift_share_threshold}, status={status}"
            )

            if status not in ("SUCCESS", "PASS", "OK"):
                drift_detected = True

        # Per-feature drift
        elif "Value Drift for column" in name:
            column = params.get("column", "UNKNOWN")
            threshold = params.get("threshold")

            m = _DRIFT_SCORE_RE.search(description)
            drift_score = float(m.group(1)) if m else None

            logger.info(
                f"{column}: drift_score={drift_score}, "
                f"threshold={threshold}, status={status}"
            )

            if status not in ("SUCCESS", "PASS", "OK"):
                drift_detected = True

    return drift_detected


def main(out_path: str) -> int:
    drift_detected = has_data_drift(reference_df, current_df)  # your existing logic
    logger.info(f"drift_detected={drift_detected}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("true" if drift_detected else "false")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Write 'true'/'false' drift result here")
    args = parser.parse_args()
    raise SystemExit(main(args.out))