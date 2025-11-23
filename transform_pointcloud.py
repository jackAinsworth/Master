#!/usr/bin/env python3
import argparse
import os
import pickle
import pandas as pd
import numpy as np

def batch_pointcloud_surfaces(input_csv,
                              output_dir,
                              output_csv,
                              surfaces_per_batch=1000,
                              start_idx=0):
    # Read input CSV of dataset pickle filepaths
    df = pd.read_csv(input_csv)
    if 'filepath' not in df.columns:
        raise ValueError("Input CSV must contain a 'filepath' column.")

    os.makedirs(output_dir, exist_ok=True)
    batch_files = []
    batch_entries = []            # now storing full entry dict
    batch_idx = start_idx
    total_surfaces = 0

    def flush_batch():
        nonlocal batch_idx, batch_entries, batch_files
        if not batch_entries:
            return
        batch_fname = os.path.join(output_dir,
                                   f"surface_batch_{batch_idx:03d}.pkl")
        with open(batch_fname, 'wb') as f:
            # dump list of dict entries
            pickle.dump(batch_entries, f)
        batch_files.append(batch_fname)
        batch_idx += 1
        batch_entries.clear()

    # Iterate over each dataset pickle
    for path in df['filepath']:
        with open(path, 'rb') as f:
            ds = pickle.load(f)
        # `ds` is a dict with key 'data' holding list of entries
        entries = ds.get('data') or ds.get('pointclouds')
        if entries is None:
            raise KeyError(f"No 'data' or 'pointclouds' in {path}")

        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Each item in 'data' must be a dict")

            # 1) Flatten the UV‐grid into a true pointcloud:
            pts = entry.get('rotated_noisy_points')
            if pts is None:
                raise KeyError("Entry missing 'rotated_noisy_points'")
            # pts originally shape (U, V, 3) or (num_u, num_v, 3)
            flat_pts = np.reshape(pts, (-1, 3))
            entry['rotated_noisy_points'] = flat_pts

            # 2) (Optionally) you could flatten the control net too,
            #    but typically it's already a small grid; keep it as-is.

            batch_entries.append(entry)
            total_surfaces += 1

            # 3) Flush when full
            if len(batch_entries) >= surfaces_per_batch:
                flush_batch()

    # Flush any remainder
    flush_batch()

    # Write a CSV that lists all the batch files
    batch_df = pd.DataFrame({'batch_filepath': batch_files})
    batch_df.to_csv(output_csv, index=False)

    print(f"Created {len(batch_files)} batch files, totaling {total_surfaces} surfaces")
    print(f"Batch manifest written to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch dataset .pkl entries into surface‐flattened pointcloud batches"
    )
    parser.add_argument("--input_csv", "-i", required=True,
                        help="CSV with 'filepath' column listing dataset .pkl files")
    parser.add_argument("--output_dir", "-d", required=True,
                        help="Directory to save batch pickle files")
    parser.add_argument("--output_csv", "-o", required=True,
                        help="CSV to write listing of batch pickle filepaths")
    parser.add_argument("--surfaces_per_batch", "-b", type=int, default=1000,
                        help="How many surfaces per batch file")
    parser.add_argument("--start_idx", "-s", type=int, default=0,
                        help="Starting batch index offset (for numbering files)")
    args = parser.parse_args()

    batch_pointcloud_surfaces(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        output_csv=args.output_csv,
        surfaces_per_batch=args.surfaces_per_batch,
        start_idx=args.start_idx
    )
