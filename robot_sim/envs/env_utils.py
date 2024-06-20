import trimesh
import numpy as np
from typing import List
import sys
import os

trimesh.util.log.setLevel('ERROR')


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_point_cloud_from_meshes(meshes: List[trimesh.Trimesh], sample_radius: float = 5e-3) -> np.ndarray:
    sample_area = np.pi * sample_radius ** 2
    cloud = []
    for i, mesh in enumerate(meshes):
        mesh_cloud = trimesh.sample.sample_surface(mesh, int(mesh.area / sample_area))[0]
        cloud.append(mesh_cloud)
    return np.concatenate(cloud, axis=0)
