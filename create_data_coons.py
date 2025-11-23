#!/usr/bin/env python3
"""
create_data_batches.py

This script creates a large dataset in batches using the DatasetGeneratorKoons module.
Each batch is saved as a separate file under a designated folder. After all batches
are generated, a CSV listing each batch file’s path is written to the logs directory.


nohup python create_data_coons.py > logs/create_coons.log 2>&1 &
"""

import os
import csv
import time
import json
import numpy as np
from datetime import datetime
import DatasetGeneratorKoons  # make sure this module is in your PYTHONPATH

# --- CONSTANTS / DIRECTORIES / LOG LOCATIONS (ALL CAPS) ---
SAVE_DIR             = "dataset_coons"
LOG_DIR              = "logs"
BATCH_FILEPATHS_CSV  = os.path.join(SAVE_DIR, "batch_filepaths.csv")
BATCH_LOG_FILE       = os.path.join(LOG_DIR, "batch_creation_1_000_000.log")

BATCH_SIZE           = 2000 #250_000
TOTAL_ENTRIES        = 2000 #1_000_000
NUM_BATCHES          = TOTAL_ENTRIES // BATCH_SIZE
FILENAME_PREFIX_BASE = "coons_dataset_batch"  # will append batch index to this

# Configuration parameters to forward to create_dataset_entry
NUM_POINTS_RANGE    = (3, 10)
PLANE_WIDTH_RANGE   = (30, 60)
PLANE_HEIGHT_RANGE  = (15, 30)
PLANE_DEPTH_RANGE   = (5, 15)
NUM_SAMPLES         = 50
NUM_V               = 50
DEGREE_U            = 3
DEGREE_V            = 3
CTRLOPTS_U_RANGE    = (4, 12)
CTRLOPTS_V_RANGE    = (4, 12)
NOISE_STD           = 0.0
NOISE_MEAN          = 0.0
RANDOM_ROTATION     = True
ANGLE_RANGES_DEG    = {
    'x': (-45, 45),
    'y': (-45, 45),
    'z': (-10, 10)
}
NORMALIZE           = True
RESAMPLE            = False
TARGET_SHAPE        = (50, 50, 3)
SHOW_VISU           = False

# ------------------------------------------------------------
def append_to_batch_log(batch_idx: int, filepath: str, elapsed: float):
    """
    Append a line to the rolling batch log containing:
      TIMESTAMP, batch index, output filepath, elapsed seconds
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"{timestamp} | Batch {batch_idx}/{NUM_BATCHES} | {filepath} | {elapsed:.2f}s"
    with open(BATCH_LOG_FILE, "a") as lf:
        lf.write(log_line + "\n")

def main():
    # Ensure directories exist
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # If the batch log already exists, do not overwrite—just append.
    # Otherwise, create it and write a header.
    if not os.path.exists(BATCH_LOG_FILE):
        with open(BATCH_LOG_FILE, "w") as lf:
            lf.write("TIMESTAMP | BATCH_IDX/TOTAL | FILEPATH | DURATION_SEC\n")

    batch_filepaths = []

    for batch_idx in range(1, NUM_BATCHES + 1):
        start_time = time.time()

        # Construct filename prefix for this batch
        filename_prefix = f"{FILENAME_PREFIX_BASE}{batch_idx}"

        # Call create_and_save_dataset for this batch
        filepath = DatasetGeneratorKoons.create_and_save_dataset(
            num_entries=BATCH_SIZE,
            save_dir=SAVE_DIR,
            filename_prefix=filename_prefix,

            # Forwarded dataset-entry parameters:
            num_points_range=NUM_POINTS_RANGE,
            plane_width_range=PLANE_WIDTH_RANGE,
            plane_height_range=PLANE_HEIGHT_RANGE,
            plane_depth_range=PLANE_DEPTH_RANGE,
            num_samples=NUM_SAMPLES,
            num_v=NUM_V,
            degree_u=DEGREE_U,
            degree_v=DEGREE_V,
            ctrlpts_size_u_range=CTRLOPTS_U_RANGE,
            ctrlpts_size_v_range=CTRLOPTS_V_RANGE,
            noise_std=NOISE_STD,
            noise_mean=NOISE_MEAN,
            random_rotation=RANDOM_ROTATION,
            angle_ranges_deg=ANGLE_RANGES_DEG,
            normalize=NORMALIZE,
            resample=RESAMPLE,
            target_shape=TARGET_SHAPE,
            show_visu=SHOW_VISU
        )

        end_time = time.time()
        elapsed = end_time - start_time

        print(f"[Batch {batch_idx}/{NUM_BATCHES}] Saved to: {filepath}  (took {elapsed:.2f}s)")
        batch_filepaths.append(filepath)

        # Append this batch’s details to the rolling log
        append_to_batch_log(batch_idx, filepath, elapsed)

    # After all batches are complete, write CSV of filepaths
    with open(BATCH_FILEPATHS_CSV, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filepath"])
        for fp in batch_filepaths:
            writer.writerow([fp])

    print(f"\nAll {NUM_BATCHES} batches created.")
    print(f"Batch filepaths CSV written to: {BATCH_FILEPATHS_CSV}")
    print(f"Batch creation log appended to: {BATCH_LOG_FILE}")

if __name__ == "__main__":
    main()
