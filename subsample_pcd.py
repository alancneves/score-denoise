#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Subsample original pointclouds to build datasets, using the
farthest point sampling (FPS) algorithm.
"""

import argparse
from pathlib import Path
from tqdm import tqdm
from glob import glob
from utils.misc import int_list
from models.utils import farthest_point_sampling_simple as fps
from datasets.pcl import load_pcd, save_pcd


__author__ = "Alan Neves"
__version__ = "1.0.0"


def downsample(input_path: str, num_pts: int):
    pcl = load_pcd(input_path)
    if len(pcl) > num_pts:
        pcl, _ = fps(pcl, num_pts)
        pcl = pcl.numpy()
    return pcl


def get_filelist(input_path: str) -> list:
    filelist = None
    if Path(input_path).is_dir():
        filelist = []
        allowed_exts = ['txt','xyz','npy','npz','ply']
        for ext in allowed_exts:
            filelist.extend( glob(f"{input_path}/**/*.{ext}", recursive=True) )
    elif Path(input_path).is_file():
        filelist = [input_path]
    else:
        raise FileNotFoundError("File not found!")
    return filelist


if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description='Subsample pointclouds using the farthest point sampling (FPS)')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input. Can be a file or a directory. If a directory, the code will get recursively all files.')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Output dir')
    parser.add_argument('-r', '--resolutions', type=int_list, default=['10000', '30000', '50000'], help='Desired number of points')
    args = parser.parse_args()

    # Load and downsample PCL
    filelist = get_filelist(args.input)
    for res in args.resolutions:
        for filename in tqdm(filelist, desc=f"Converting PCLs to {res} points"):
            new_pcl = downsample(filename, int(res))
            new_pcl_path = Path(args.output_dir) / f"{Path(filename).stem}_{res}{Path(filename).suffix}"
            save_pcd(new_pcl_path, new_pcl)
