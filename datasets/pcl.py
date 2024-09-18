import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from pathlib import Path
from plyfile import PlyData, PlyElement
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


def save_pcd(filepath: str, pcl: np.array):
    # Create subjacent folders
    Path(filepath).parent.mkdir(exist_ok=True, parents=True)
    # Create filetype as choosed by user
    ext = Path(filepath).suffix
    if ext == '.txt' or ext == '.xyz':
        np.savetxt(filepath, pcl, fmt='%.8f')
    elif ext == '.npy' or ext == '.npz':
        np.save(filepath, pcl)
    elif ext == '.ply':
        vertex = np.zeros(len(pcl), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        vertex['x'] = pcl[:,0]
        vertex['y'] = pcl[:,1]
        vertex['z'] = pcl[:,2]
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el]).write(filepath)
    else:
        raise NotImplementedError(f"Files with ext {ext} are not allowed to be created!")
    