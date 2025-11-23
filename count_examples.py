#!/usr/bin/env python3
# count_tfrecords.py
import argparse, glob, os, sys
import tensorflow as tf

def list_inputs(patterns):
    files = []
    for p in patterns:
        if os.path.isdir(p):
            files.extend(glob.glob(os.path.join(p, "*.tfrecord")))
        else:
            files.extend(glob.glob(p))
    files = sorted(set(files))
    if not files:
        print("No TFRecord files matched the given paths/patterns.", file=sys.stderr)
        sys.exit(2)
    return files

def count_file(path, compression=None):
    ds = tf.data.TFRecordDataset(path, compression_type=(compression or ""))
    # Use a dataset reduction so it streams efficiently
    return int(ds.reduce(tf.constant(0, dtype=tf.int64), lambda c, _: c + 1).numpy())

def main():
    ap = argparse.ArgumentParser(description="Count examples in TFRecord file(s).")
    ap.add_argument("paths", nargs="+",
                    help="File(s), dir(s), or glob(s). E.g. /home/ainsworth/master/tfrecords_coons/data_*-of-*.tfrecord")
    ap.add_argument("--compression", choices=["GZIP", "ZLIB"], default=None,
                    help="Set if your TFRecords were written with compression.")
    ap.add_argument("--per-file", action="store_true", help="Print per-file counts.")
    args = ap.parse_args()

    files = list_inputs(args.paths)
    total = 0
    for f in files:
        n = count_file(f, args.compression)
        if args.per_file:
            print(f"{f}\t{n}")
        total += n
    if len(files) > 1:
        print(f"TOTAL\t{total}")

if __name__ == "__main__":
    main()
