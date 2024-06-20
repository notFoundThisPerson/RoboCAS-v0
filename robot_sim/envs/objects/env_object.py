import sapien.core as sapien
from dataclasses import dataclass
from typing import Union, List, Optional, Dict
import numpy as np
import os
import trimesh

from robot_sim.utils import format_path
from robot_sim.envs.env_utils import get_point_cloud_from_meshes
from mani_skill2.utils.trimesh_utils import get_articulation_meshes, merge_meshes


@dataclass
class BaseObjectConfig:
    urdf_path: Optional[str] = None
    visual_path: Optional[str] = None
    collision_path: Optional[str] = None
    name: Optional[str] = None
    initial_pose: sapien.Pose = sapien.Pose()
    model_origin_offset: Optional[sapien.Pose] = None
    scale: Union[float, List[float]] = 1.0
    density: float = 1000.0
    use_fixed_base: bool = False
    use_nonconvex_collision: bool = False
    physical_material: Dict[str, float] = None
    render_material: Dict[str, float] = None


class BaseObject:
    def __init__(self, scene: sapien.Scene, config: BaseObjectConfig):
        self._scene = scene
        self.config = config

        if config.urdf_path is not None:
            self.urdf_path = format_path(config.urdf_path)
            self.visual_path, self.collision_path = None, None
        else:
            self.visual_path = format_path(config.visual_path)
            assert os.path.exists(self.visual_path), 'No such file: ' + self.visual_path
            if config.collision_path is not None:
                self.collision_path = format_path(config.collision_path)
                assert os.path.exists(self.collision_path)
            else:
                self.collision_path = self.visual_path
            self.urdf_path = None
        self.name = config.name if config.name is not None else os.path.basename(self.visual_path).split('.')[0]
        self.obj_scale = np.broadcast_to(config.scale, 3)
        if config.physical_material is not None:
            self.material = self._scene.create_physical_material(**config.physical_material)
        else:
            self.material = None
        if self.config.model_origin_offset is not None:
            self.origin_offset = self.config.model_origin_offset
        else:
            self.origin_offset = sapien.Pose()
        self._load_model()

    def _load_model(self):
        if self.urdf_path is not None:
            loader = self._scene.create_urdf_loader()
            loader.fix_root_link = self.config.use_fixed_base
            loader.scale = self.config.scale
            config = {}
            if self.material is not None:
                config['material'] = self.material
            self.model = loader.load(self.urdf_path, config)
            self._ignore_collision()
        else:
            builder = self._scene.create_actor_builder()
            builder.add_multiple_collisions_from_file(filename=self.collision_path, scale=self.obj_scale,
                                                      pose=self.origin_offset, material=self.material,
                                                      density=self.config.density)
            if self.config.render_material is not None:
                render_mtl = self._scene.engine.get_renderer().create_material()
                for k, v in self.config.render_material.items():
                    setattr(render_mtl, k, v)
            else:
                render_mtl = None
            builder.add_visual_from_file(filename=self.visual_path, scale=self.obj_scale, pose=self.origin_offset,
                                         material=render_mtl)
            if self.config.use_fixed_base:
                self.model = builder.build_static(self.name)
            else:
                self.model = builder.build(self.name)
                self.model.set_damping(0.1, 0.1)
        self.set_pose(self.config.initial_pose)

    def _ignore_collision(self):
        """Ignore collision within the articulation to avoid impact from imperfect collision shapes."""
        # The legacy version only ignores collision of child links of active joints.
        for link in self.model.get_links():
            for s in link.get_collision_shapes():
                g0, g1, g2, g3 = s.get_collision_groups()
                s.set_collision_groups(g0, g1, g2 | 1 << 31, g3)

    @property
    def pose(self):
        if self.urdf_path is not None:
            return self.model.pose * self.origin_offset.inv()
        else:
            return self.model.pose

    @pose.setter
    def pose(self, pose: sapien.Pose):
        self.set_pose(pose)

    def set_pose(self, pose: sapien.Pose):
        if self.urdf_path is not None:
            self.model.set_pose(pose * self.origin_offset)
        else:
            self.model.set_pose(pose)

    def get_point_cloud(self, sample_radius=5e-3):
        obj_meshes = [self.get_obj_mesh()]
        return get_point_cloud_from_meshes(obj_meshes, sample_radius)

    def get_AABB(self, local_frame=False, local_transform: Optional[np.ndarray] = None):
        obj_mesh = self.get_obj_mesh(local_frame)
        verts = obj_mesh.vertices
        if local_frame and local_transform is not None:
            verts = np.matmul(verts, local_transform[:3, :3].T) + local_transform[np.newaxis, :3, 3]
        mins = np.min(verts, axis=0)
        maxs = np.max(verts, axis=0)
        return mins, maxs

    def get_obj_mesh(self, local_frame=False) -> trimesh.Trimesh:
        if self.urdf_path is not None:
            mesh = merge_meshes(get_articulation_meshes(self.model))
            if local_frame:
                mesh.apply_transform(self.pose.inv().to_transformation_matrix())
        else:
            mesh = trimesh.load(self.collision_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = merge_meshes(list(mesh.geometry.values()))
            mesh.apply_scale(self.obj_scale)
            if local_frame:
                mesh.apply_transform(self.origin_offset.to_transformation_matrix())
            else:
                mesh.apply_transform((self.pose * self.origin_offset).to_transformation_matrix())
        return mesh

    def get_name(self):
        return self.name

    def __repr__(self):
        return "name: %s, visual: %s, collision: %s" % (self.name, self.config.visual_path, self.config.collision_path)
