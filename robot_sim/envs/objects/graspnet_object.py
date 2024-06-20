import sapien.core as sapien
from pathlib import Path
import json
from transforms3d.euler import euler2quat
from transforms3d.quaternions import mat2quat
import numpy as np

from robot_sim.envs.objects.env_object import BaseObject, BaseObjectConfig
from robot_sim.utils import format_path


def generate_views(N, phi=(np.sqrt(5) - 1) / 2, center=np.zeros(3, dtype=np.float32), R=1):
    ''' Author: chenxi-wang
    View sampling on a sphere using Febonacci lattices.
    **Input:**
    - N: int, number of viewpoints.
    - phi: float, constant angle to sample views, usually 0.618.
    - center: numpy array of (3,), sphere center.
    - R: float, sphere radius.
    **Output:**
    - numpy array of (N, 3), coordinates of viewpoints.
    '''
    idxs = np.arange(N, dtype=np.float32)
    Z = (2 * idxs + 1) / N - 1
    X = np.sqrt(1 - Z ** 2) * np.cos(2 * idxs * np.pi * phi)
    Y = np.sqrt(1 - Z ** 2) * np.sin(2 * idxs * np.pi * phi)
    views = np.stack([X, Y, Z], axis=1)
    views = R * np.array(views) + center
    return views


def get_model_grasps(datapath):
    ''' Author: chenxi-wang
    Load grasp labels from .npz files.
    '''
    label = np.load(datapath)
    points = label['points']
    offsets = label['offsets']
    scores = label['scores']
    collision = label['collision']
    return points, offsets, scores, collision


def viewpoint_params_to_matrix(towards, angle):
    '''
    **Input:**
    - towards: numpy array towards vector with shape (N, 3).
    - angle: float of in-plane rotation.
    **Output:**
    - numpy array of the rotation matrix with shape (N, 3, 3).
    '''
    axis_x = towards
    axis_y = np.stack([-axis_x[:, 1], axis_x[:, 0], np.zeros_like(axis_x[:, 2])], axis=1)

    norm_x = np.linalg.norm(axis_x, axis=1, keepdims=True)
    norm_y = np.linalg.norm(axis_y, axis=1, keepdims=True)
    zero_flag = norm_y[:, 0] == 0
    axis_y[zero_flag] = np.array([[0, 1, 0]])
    norm_y[zero_flag] = 1
    axis_x = axis_x / norm_x
    axis_y = axis_y / norm_y
    axis_z = np.cross(axis_x, axis_y)
    R2 = np.stack([axis_x, axis_y, axis_z], axis=2)

    sin_a = np.sin(angle)
    cos_a = np.cos(angle)
    zeros = np.zeros_like(angle)
    ones = np.ones_like(angle)
    R1 = np.stack([ones, zeros, zeros, zeros, cos_a, -sin_a, zeros, sin_a, cos_a], axis=1).reshape(-1, 3, 3)

    matrix = np.matmul(R2, R1)
    return matrix.astype(np.float32)


def draw_grasps(scene: sapien.Scene, poses: np.ndarray):
    ctrl_points = np.array([[0.04, 0, 0], [0.04, 0, -0.05], [0, 0, -0.05], [0, 0, -0.1],
                            [0, 0, -0.05], [-0.04, 0, -0.05], [-0.04, 0, 0]])
    if len(poses.shape) == 2:
        poses = poses[np.newaxis]
    lines = []
    ctrl_points = np.matmul(ctrl_points[np.newaxis], poses[:, :3, :3].transpose((0, 2, 1))) + poses[:, np.newaxis, :3, 3]
    for ctrl in ctrl_points:
        for p1, p2 in zip(ctrl[:-1], ctrl[1:]):
            center = (p1 + p2) / 2
            x = p1 - p2
            length = np.linalg.norm(x)
            x /= length
            y = np.array([x[1], -x[0], 0])
            y_norm = np.linalg.norm(y)
            if y_norm == 0:
                y = np.array([0, 1, 0])
            else:
                y /= y_norm
            z = np.cross(x, y)
            mat = np.stack([x, y, z], axis=1)
            quat = mat2quat(mat)
            pose = sapien.Pose(center, quat)

            builder = scene.create_actor_builder()
            builder.add_capsule_visual(pose, 5e-3, length / 2., np.array([1, 0, 0]))
            lines.append(builder.build_static())
    return lines


class GraspNetObject(BaseObject):
    def __init__(self, scene: sapien.Scene, dataset_path: str, obj_index: str, pose: sapien.Pose = sapien.Pose()):
        self.dataset_path = Path(format_path(dataset_path))
        self.obj_index = obj_index.zfill(3)
        config = self._generate_config(pose)
        super().__init__(scene, config)
        self.local_transform = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)

    def _generate_config(self, pose):
        # visual_path = self.dataset_path / 'simplified_models/visual' / self.obj_index / 'textured_simplified.obj'
        visual_path = self.dataset_path / 'models' / self.obj_index / 'textured.obj'
        collision_path = self.dataset_path / 'simplified_models/collision' / ('%s.obj' % self.obj_index)
        obj_info_path = self.dataset_path / 'simplified_models/visual' / self.obj_index / 'obj_info.json'

        with open(obj_info_path.as_posix(), 'r') as f:
            obj_info = json.load(f)
        self.origin_offset = sapien.Pose(q=euler2quat(*obj_info['orn']))

        config = BaseObjectConfig(
            visual_path=visual_path.as_posix(),
            collision_path=collision_path.as_posix(),
            name=obj_info['name'],
            initial_pose=pose,
            model_origin_offset=sapien.Pose(q=euler2quat(*obj_info['orn'])),
            physical_material=dict(static_friction=0.99, dynamic_friction=0.99, restitution=0)
        )
        return config

    def get_grasp_labels(self):
        pre_computed_data = self.dataset_path / ('filtered_grasps/%s_labels.npz' % self.obj_index)
        if pre_computed_data.exists():
            data = np.load(pre_computed_data.as_posix())
            grasp_mats = data['grasp_mats']
            grasp_width = data['grasp_width']
            scores = data['scores']
        else:
            grasp_label_file = self.dataset_path / ('grasp_label/%s_labels.npz' % self.obj_index)
            sampled_points, offsets, scores, _ = get_model_grasps(grasp_label_file.as_posix())

            num_samples, num_views, num_angles, num_depths = scores.shape
            views = generate_views(num_views)

            th = 0.3
            max_width = 0.08
            flag = np.all(np.stack([scores <= th, scores >= 0, offsets[..., -1] <= max_width], axis=-1), axis=-1)
            offsets = offsets[flag]
            scores = 1.1 - scores[flag]

            num_grasps = offsets.shape[0]
            grasp_mats = np.eye(4)[np.newaxis].repeat(num_grasps, axis=0)

            point_indices, view_indices, _, _ = np.where(flag)
            grasp_mats[:, :3, 3] = sampled_points[point_indices]
            view_grasp = -views[view_indices]
            angle_grasp = offsets[:, 0]
            grasp_mats[:, :3, :3] = viewpoint_params_to_matrix(view_grasp, angle_grasp)

            local_tf = self.local_transform[np.newaxis].repeat(num_grasps, axis=0)
            local_tf[:, 0, 3] = offsets[:, 1]
            grasp_mats = np.matmul(grasp_mats, local_tf)
            grasp_width = offsets[:, 2]

            pre_computed_data.parent.mkdir(exist_ok=True)
            np.savez(pre_computed_data.as_posix(), grasp_mats=grasp_mats, grasp_width=grasp_width, scores=scores)
        grasp_mats = np.matmul(self.origin_offset.to_transformation_matrix()[np.newaxis], grasp_mats)
        return grasp_mats, grasp_width, scores

    def get_grasps_in_cur_scene(self):
        grasps, widths, scores = self.get_grasp_labels()
        obj_pose = self.model.pose.to_transformation_matrix()
        grasps = np.matmul(obj_pose[np.newaxis], grasps)
        return grasps, widths, scores
