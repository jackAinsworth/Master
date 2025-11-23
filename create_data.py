#!/usr/bin/env python3
"""
create_data.py

This script creates a dataset using the DatasetGenerator module with specified parameters.
It logs the execution time, the configuration parameters, and the output file path to a log file
under the 'logs' directory.

nohup python create_data.py > logs/create_ruled.log 2>&1 &

"""

import os
import time
import json
import numpy as np
import DatasetGenerator  # make sure this module is in your PYTHONPATH

# --- Configuration parameters ---
config = {
    "num_entries": 500_000,             # 120000 40000 2000000
    "num_points_range": [3, 10],
    "plane_width_range": [20, 100],
    "plane_height_range": [20, 60],
    "num_samples": 50,
    "num_v": 50,
    "noise_std": 0.0,
    "noise_mean": 0.75,
    "degree_u": 3,
    "degree_v": 3,
    "ctrlpts_size_u_range": [4, 12],
    "ctrlpts_size_v_range": [4, 12],
    "z_gap_range": [20, 80],
    "bspline_degree": 3,
    "extrapolate": False,
    "random_rotation": True,
    "show_visu": False
}

# Convert list parameters to tuples for consistency.
config["num_points_range"] = tuple(config["num_points_range"])
config["plane_width_range"] = tuple(config["plane_width_range"])
config["plane_height_range"] = tuple(config["plane_height_range"])
config["ctrlpts_size_u_range"] = tuple(config["ctrlpts_size_u_range"])
config["ctrlpts_size_v_range"] = tuple(config["ctrlpts_size_v_range"])
config["z_gap_range"] = tuple(config["z_gap_range"])

# --- End configuration ---

def main():
    start_time = time.time()

    # Create and save the dataset using the provided configuration
    dataset_file = DatasetGenerator.create_and_save_dataset(**config)

    end_time = time.time()
    elapsed = end_time - start_time

    # Prepare log output including the configuration parameters
    log_output = []
    log_output.append("Dataset saved to: {}\n".format(dataset_file))
    log_output.append("Execution time: {:.2f} seconds\n".format(elapsed))
    log_output.append("Configuration Parameters:\n")

    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    config_str = json.dumps(config, indent=4, default=convert)
    log_output.append(config_str)

    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, "dataset_executionbetterapprox.log")

    with open(log_file, "w") as f:
        f.write("\n".join(log_output))

    print("Dataset saved to:", dataset_file)
    print("Execution time: {:.2f} seconds".format(elapsed))
    print("Configuration parameters logged to:", log_file)

if __name__ == '__main__':
    main()
