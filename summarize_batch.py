import pickle

def summarize_batch(path, max_items=5):
    """
    Load a batch .pkl and print a summary of its contents.
    - path:       filesystem path to e.g. "surface_batch_000.pkl"
    - max_items:  how many individual entries to print details for
    """
    with open(path, "rb") as f:
        batch = pickle.load(f)

    print(f"Loaded batch from {path!r}")
    print(f"Type of `batch`: {type(batch)}")
    try:
        length = len(batch)
        print(f"Number of entries in batch: {length}")
    except TypeError:
        print("  (not a sized collection)")

    # If it's a list or similar, inspect a few entries
    if hasattr(batch, "__iter__"):
        for i, entry in enumerate(batch):
            if i >= max_items:
                print(f"... ({length - max_items} more entries)")
                break
            print(f"\nEntry #{i}: type={type(entry)}")
            # If it's a NumPy array
            try:
                import numpy as np
                if isinstance(entry, np.ndarray):
                    print(f"  NumPy array, shape={entry.shape}, dtype={entry.dtype}")
                    continue
            except ImportError:
                pass
            # If it's a dict (with rotated_noisy_points, rotated_control_net, etc.)
            if isinstance(entry, dict):
                keys = list(entry.keys())
                print(f"  dict with keys: {keys}")
                for k in keys:
                    v = entry[k]
                    if hasattr(v, "shape"):
                        print(f"    {k}: type={type(v)}, shape={getattr(v, 'shape', None)}")
                    else:
                        print(f"    {k}: type={type(v)}")
                continue
            # Fallback: just print repr summary
            print(f"  repr(entry)[:200]: {repr(entry)[:200]}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python summarize_batch.py surface_batch_000.pkl")
        sys.exit(1)
    summarize_batch(sys.argv[1])
