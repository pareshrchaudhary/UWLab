# Copyright (c) 2024-2025, The UW Lab Project Developers.
# Proprietary and Confidential - All Rights Reserved.

"""Verify collected demos from collect_demos.py (zarr or HDF5)."""

import argparse
import numpy as np


def verify_zarr(path: str) -> None:
    import zarr
    root = zarr.open(path, mode="r")
    if "meta" not in root or "episode_ends" not in root["meta"]:
        print(f"Invalid dataset: missing meta/episode_ends in {path}")
        return
    ends = np.asarray(root["meta"]["episode_ends"])
    num_episodes = len(ends)
    total_frames = int(ends[-1]) if num_episodes > 0 else 0
    print(f"Dataset: {path}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Total frames: {total_frames}")
    if num_episodes > 0:
        lengths = np.diff(ends, prepend=0)
        print(f"  Episode lengths: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.1f}")
    data = root.get("data")
    if data is None:
        print("  No 'data' group.")
        return
    print("  Keys:", list(data.keys()))
    for key in list(data.keys())[:10]:
        arr = data[key]
        if hasattr(arr, "shape"):
            print(f"    data/{key}: shape={arr.shape}, dtype={arr.dtype}")
        else:
            for subk in list(arr.keys())[:3]:
                a = arr[subk]
                print(f"    data/{key}/{subk}: shape={a.shape}, dtype={a.dtype}")
    if num_episodes > 0 and "actions" in data:
        actions = np.asarray(data["actions"])
        end0 = int(ends[0])
        sample = actions[: min(3, end0)]
        print(f"  Actions sample (first episode, first steps):\n{sample}")


def verify_hdf5(path: str) -> None:
    import h5py
    with h5py.File(path, "r") as f:
        if "meta" not in f or "episode_ends" not in f["meta"]:
            print(f"Invalid dataset: missing meta/episode_ends in {path}")
            return
        ends = f["meta"]["episode_ends"][:]
        num_episodes = len(ends)
        total_frames = int(ends[-1]) if num_episodes > 0 else 0
        print(f"Dataset: {path}")
        print(f"  Episodes: {num_episodes}")
        print(f"  Total frames: {total_frames}")
        if num_episodes > 0:
            lengths = np.diff(ends, prepend=0)
            print(f"  Episode lengths: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.1f}")
        data = f.get("data")
        if data is not None:
            print("  Keys:", list(data.keys()))
            for key in list(data.keys())[:10]:
                arr = data[key]
                if isinstance(arr, h5py.Dataset):
                    print(f"    data/{key}: shape={arr.shape}, dtype={arr.dtype}")
                else:
                    for subk in list(arr.keys())[:3]:
                        a = arr[subk]
                        print(f"    data/{key}/{subk}: shape={a.shape}, dtype={a.dtype}")


def main():
    parser = argparse.ArgumentParser(description="Verify demos collected by collect_demos.py")
    parser.add_argument("dataset_file", type=str, help="Path to dataset (e.g. ./datasets/dataset.zarr)")
    args = parser.parse_args()
    path = args.dataset_file
    if path.endswith(".zarr"):
        verify_zarr(path)
    elif path.endswith(".hdf5") or path.endswith(".h5"):
        verify_hdf5(path)
    else:
        print("Unknown format. Use .zarr or .hdf5/.h5")
        return
    print("Done.")


if __name__ == "__main__":
    main()
