#!/usr/bin/env python3
"""Experiment 006-B: 100 new HDF5, fixed stratified 80/20 split, ACT from scratch."""
from __future__ import annotations
import argparse
import json
import os
import runpy
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--rmb-root", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=25)
    p.add_argument("--skip", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Python 3.12 is required')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required')

    dataset = args.dataset_dir.resolve(strict=True)
    manifest = json.loads(args.split_json.read_text(encoding="utf-8"))
    train = list(manifest["train"])
    validation = list(manifest["validation"])
    if len(train) != 80 or len(validation) != 20:
        raise ValueError('Expected 80 train and 20 validation episodes')
    if len(set(train + validation)) != 100 or (set(train) & set(validation)):
        raise ValueError('Overlapping or duplicate episode names')
    all_hdf5 = {f.name for f in dataset.glob('*.hdf5')}
    if all_hdf5 != set(train + validation):
        raise ValueError('Dataset must contain exactly the 100 split files')

    ordered_files = [str((dataset / n).resolve(strict=True)) for n in train + validation]
    rmb_root = args.rmb_root.resolve(strict=True)
    entry = rmb_root / 'bin' / 'Train.py'
    if not entry.is_file():
        raise FileNotFoundError(entry)
    checkpoint = args.checkpoint_dir.resolve()
    if checkpoint.exists() and any(checkpoint.iterdir()):
        raise FileExistsError(f'Checkpoint directory is not empty: {checkpoint}')

    sys.path.insert(0, str(rmb_root.parent))
    from robo_manip_baselines.common.base.TrainBase import TrainBase

    def fixed_files(self):
        self.all_filenames = ordered_files.copy()
        print('Experiment 006-B fixed split: train=80 validation=20', flush=True)
        print('First training:', Path(ordered_files[0]).name, flush=True)
        print('First validation:', Path(ordered_files[80]).name, flush=True)

    original_stats = TrainBase.set_data_stats
    def train_only_stats(self):
        original_files = self.all_filenames
        try:
            self.all_filenames = original_files[:80]
            print('Normalization: training 80 episodes ONLY', flush=True)
            return original_stats(self)
        finally:
            self.all_filenames = original_files

    TrainBase.setup_rmb_files = fixed_files
    TrainBase.set_data_stats = train_only_stats
    sys.argv = [
        str(entry), 'Act',
        '--dataset_dir', str(dataset),
        '--checkpoint_dir', str(checkpoint),
        '--state_keys', 'measured_joint_pos',
        '--action_keys', 'command_joint_pos',
        '--camera_names', 'front',
        '--train_ratio', '0.8',
        '--num_epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--chunk_size', str(args.chunk_size),
        '--skip', str(args.skip),
        '--num_workers', str(args.num_workers),
        '--seed', '42',
        '--no-enable_rmb_cache',
    ]
    print('Fresh training: no pretrained ACT checkpoint', flush=True)
    runpy.run_path(str(entry), run_name='__main__')

if __name__ == '__main__':
    main()
