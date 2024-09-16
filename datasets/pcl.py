import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from pathlib import Path
from plyfile import PlyData
from glob import glob

class PointCloudDataset(Dataset):

    def __init__(self, pcl_dir, transform=None):
        super().__init__()
        self.pcl_dir = pcl_dir
        self.transform = transform
        self.pointclouds = []
        self.pointcloud_names = []
        for fn in tqdm(glob(f"{self.pcl_dir}/**/*", recursive=True)):
            pcl_path = os.path.join(self.pcl_dir, fn)
            pcl = torch.FloatTensor(load_pcd(pcl_path), device='cpu')
            self.pointclouds.append(pcl)
            self.pointcloud_names.append(fn[:-4])

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {
            'pcl_clean': self.pointclouds[idx].clone(), 
            'name': self.pointcloud_names[idx]
        }
        if self.transform is not None:
            data = self.transform(data)
        return data


def load_pcd(filepath: str) -> np.array:
    data = None
    ext = Path(filepath).suffix
    if not Path(filepath).is_file():
        raise FileNotFoundError(f"File {filepath} not found!")

    if ext == '.txt' or ext == '.xyz':
        data = np.genfromtxt(filepath, dtype=np.float32)
    elif ext == '.npy' or ext == '.npz':
        data = np.load(filepath).as_type(np.float32)
    elif ext == '.ply':
        plydata = PlyData.read(filepath)
        data = np.array(plydata['vertex'].data[['x','y','z']].tolist(), dtype=np.float32)
    else:
        raise NotImplementedError(f"Files with ext {ext} are not allowed!")

    return data