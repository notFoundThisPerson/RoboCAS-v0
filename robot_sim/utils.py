from pathlib import Path
import numpy as np
from typing import Union, Dict, List
import sapien.core as sapien
from enum import Enum

from mani_skill2 import PACKAGE_DIR, PACKAGE_ASSET_DIR, ASSET_DIR

MT_ASSET_DIR = (Path(__file__).parents[1] / 'assets').as_posix()


def format_path(s: str):
    return s.format(PACKAGE_DIR=PACKAGE_DIR, PACKAGE_ASSET_DIR=PACKAGE_ASSET_DIR,
                    ASSET_DIR=ASSET_DIR, MT_ASSET_DIR=MT_ASSET_DIR)


def create_default_world():
    import sapien.core as sapien
    from sapien.utils.viewer import Viewer

    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    scene = engine.create_scene(scene_config)
    scene.set_timestep(1 / 240.0)

    render_material = engine.get_renderer().create_material()
    render_material.set_diffuse_texture_from_file(MT_ASSET_DIR + '/textures/ground_textures/wooden_floor.jpeg')
    scene.add_ground(0, render_material=render_material)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=2, y=0, z=1)
    viewer.set_camera_rpy(r=0, p=-0.3, y=np.pi)
    return engine, scene, viewer


def get_nearest_unique_obj(obj_list, position):
    output = {}
    for idx, obj in enumerate(obj_list):
        dist = np.linalg.norm(obj.pose.p - position)
        if obj.name not in output or output[obj.name][1] > dist:
            output[obj.name] = (obj, dist, idx)
    return [v[0] for v in output.values()], [v[2] for v in output.values()]


def rotation_geodesic_error(rotation1: Union[np.ndarray, sapien.Pose], rotation2: Union[np.ndarray, sapien.Pose]):
    def uniform_format(rotation):
        if isinstance(rotation, sapien.Pose):
            rotation = rotation.to_transformation_matrix()
        return rotation[:3, :3]

    rotation1 = uniform_format(rotation1)
    rotation2 = uniform_format(rotation2)
    return np.arccos(np.clip((np.trace(np.matmul(rotation1, rotation2.T)) - 1) / 2., -1, 1))


def get_pose_error(pose1: Union[np.ndarray, sapien.Pose], pose2: Union[np.ndarray, sapien.Pose]):
    def get_position(pose):
        if isinstance(pose, sapien.Pose):
            return pose.p
        elif pose.shape == (3, 3):
            return np.zeros(3)
        return pose[:3, 3]

    pos1 = get_position(pose1)
    pos2 = get_position(pose2)
    return np.linalg.norm(pos1 - pos2), rotation_geodesic_error(pose1, pose2)


def create_fixed_constraint(link1: sapien.ActorBase, link2: sapien.ActorBase):
    return TemporaryFixedConstraint(link1, link2)


class TemporaryFixedConstraint:
    def __init__(self, link1: sapien.ActorBase, link2: sapien.ActorBase):
        self.scene = link1.get_scene()
        self.link1 = link1
        self.link2 = link2
        self.build()

    def build(self):
        rel_pose = self.link1.pose.inv() * self.link2.pose
        self.constraint = self.scene.create_drive(self.link1, rel_pose, self.link2, sapien.Pose())
        motion_constraint = [True] * 6
        self.constraint.lock_motion(*motion_constraint)

    def release(self):
        self.scene.remove_drive(self.constraint)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class DataLog:
    def __init__(self):
        self.data = None
        self.length = 0

    def append(self, new_data: Dict):
        self.length += 1
        if self.data is None:
            def create_data(src, dst):
                for k, v in src.items():
                    if isinstance(v, np.ndarray):
                        dst[k] = v[np.newaxis]
                    elif isinstance(v, Dict):
                        dst[k] = v.__class__()
                        create_data(v, dst[k])
                    else:
                        dst[k] = np.array([v])

            self.data = new_data.__class__()
            create_data(new_data, self.data)
        else:
            def contact_data(src, dst):
                for k, v in src.items():
                    if k not in dst:
                        continue
                    if isinstance(v, np.ndarray):
                        dst[k] = np.concatenate([dst[k], v[np.newaxis]], 0)
                    elif isinstance(v, Dict):
                        contact_data(v, dst[k])
                    else:
                        dst[k] = np.concatenate([dst[k], np.array([v])], 0)

            contact_data(new_data, self.data)

    def extend(self, other):
        self.length += len(other)
        if other.data is None:
            return
        if self.data is None:
            self.data = other.data
            return

        def extend_data(src, dst):
            for k, v in src.items():
                if k not in dst:
                    continue
                if isinstance(v, np.ndarray):
                    dst[k] = np.concatenate([dst[k], v], 0)
                else:
                    assert isinstance(v, Dict)
                    extend_data(v, dst[k])

        extend_data(other.data, self.data)

    def clear(self):
        self.data = None
        self.length = 0

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        if self.data is None:
            self.data = dict()
        self.data[key] = value

    def __len__(self):
        return self.length

    @property
    def last_log(self):
        if self.data is None:
            return {}

        def get_last_data(src: Dict, dst: Dict):
            for k, v in src.items():
                if isinstance(v, Dict):
                    dst[k] = v.__class__()
                    get_last_data(v, dst[k])
                else:
                    dst[k] = v[-1]

        last_log = {}
        get_last_data(self.data, last_log)
        return last_log

    @classmethod
    def from_list(cls, data: List):
        obj = cls()
        if len(data) > 0:
            obj.length = len(data)
            obj.data = data[0].__class__()
            path = []

            def get_data_from_path(d: Dict, p: list):
                if len(p) == 0:
                    return d
                return get_data_from_path(d[p[0]], p[1:])

            def merge_data(dst):
                v = get_data_from_path(data[0], path)
                if isinstance(v, Dict):
                    dst[path[-1]] = v.__class__()
                    for key in v.keys():
                        path.append(key)
                        merge_data(dst[path[-2]])
                        path.pop(-1)
                elif isinstance(v, np.ndarray):
                    dst[path[-1]] = np.stack([get_data_from_path(d, path) for d in data], 0)
                else:
                    dst[path[-1]] = np.array([get_data_from_path(d, path) for d in data])

            for key in data[0].keys():
                path.append(key)
                merge_data(obj.data)
                path.pop(-1)
        return obj


class GoalTypes(Enum):
    MOVE_AND_GRASP = 'move and grasp'
    GRASP_OBJECT = 'grasp object'
    RELEASE_OBJECT = 'release object'
    LIFT_OBJECT = 'lift object'
    TRANSFER_OBJECT = 'transfer object'
    GRASP_AND_LIFT = 'grasp and lift the object'
    OPERATE_OBJECT = 'operate the target'
    FIND_OBJECT = 'find the object'


def extend_goal_to_log(log: DataLog, lang_goal: str, goal_type: GoalTypes):
    if len(log) > 0:
        log['task'] = dict(
            lang_goal=np.array([lang_goal] * len(log)),
            goal_type=np.array([goal_type.value] * len(log))
        )
    return log
