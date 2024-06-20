import sapien.core as sapien
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
import logging
import typing

from robot_sim.envs.objects.env_object import BaseObject, BaseObjectConfig


@dataclass
class OperationConfig:
    name: str                                   # Name of the task
    type: str                                   # Type of the task
    link_name: str                              # Name of target link
    joint_name: str                             # Corresponding joint, to get state of the operation
    # Relative pose of the gripper wrt target link, z-axis for approaching direction, x-axis for finger closing direction
    handle_pose: sapien.Pose = sapien.Pose()


@dataclass
class ArticulationObjectConfig(BaseObjectConfig):
    use_fixed_base: bool = True
    surfaces: OrderedDict = None                               # Under object coordinates
    operation_tasks: OrderedDict[str, OperationConfig] = None  # Available operations on this object
    initial_joint_status: OrderedDict[str, float] = None       # Initial state of the joint


class ArticulationObject(BaseObject):
    def __init__(self, scene: sapien.Scene, config: ArticulationObjectConfig, available_layers=None):
        if not config.use_fixed_base:
            logging.info('Warning: fix the base of the articulation is recommended')
        super().__init__(scene, config)
        self.config = typing.cast(ArticulationObjectConfig, self.config)
        self._surfaces = config.surfaces if config.surfaces is not None else OrderedDict()
        self._available_layers = available_layers
        self.set_joints_properties()

    def set_joints_properties(self):
        if isinstance(self.model, sapien.Articulation):
            joint_init_qpos = []
            for joint in self.model.get_active_joints():
                joint.set_friction(np.random.uniform(0.05, 0.15))
                joint.set_drive_property(stiffness=0, damping=np.random.uniform(5, 20))
                joint_init_qpos.append(joint.get_limits()[0, 0])
            self.model.set_qpos(joint_init_qpos)

    @property
    def surfaces(self):
        if self._available_layers is not None:
            surfaces = OrderedDict()
            for layer in self._available_layers:
                if layer not in self._surfaces.keys():
                    continue
                surfaces[layer] = self._surfaces[layer]
            return surfaces
        return self._surfaces

    @surfaces.setter
    def surfaces(self, surfs: OrderedDict):
        for v in surfs.values():
            assert len(v) == 2 and len(v[0]) == 3 and len(v[1]) == 3
        self._surfaces = surfs

    def sample_place_position(self, layer=None):
        if layer is None:
            layer = np.random.choice(list(self.surfaces.keys()), 1)[0]
        else:
            assert layer in self.surfaces.keys()
        pos = np.random.uniform(self._surfaces[layer][0], self._surfaces[layer][1])
        pose = sapien.Pose(pos)
        return self.pose * pose

    @property
    def operate_tasks(self):
        return list(self.config.operation_tasks.values()) if self.config.operation_tasks is not None else []
