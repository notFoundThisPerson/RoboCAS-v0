from dataclasses import dataclass
from typing import Sequence, Union
from scipy.spatial.transform import Rotation
import sapien.core as sapien

from mani_skill2.utils.sapien_utils import get_entity_by_name
from mani_skill2.agents.controllers.pd_ee_pose import PDEEPosController, PDEEPoseController
from mani_skill2.agents.base_controller import ControllerConfig


class PDEEPosMobileController(PDEEPosController):
    config: "PDEEPosMobileControllerConfig"

    def _initialize_joints(self):
        super()._initialize_joints()
        if self.config.arm_base_link:
            self.arm_base_link = get_entity_by_name(
                self.articulation.get_links(), self.config.arm_base_link
            )
        else:
            # The child link of last joint is assumed to be the end-effector.
            self.arm_base_link = self.joints[0].get_parent_link()
        self.arm_base_link_idx = self.articulation.get_links().index(self.arm_base_link)

    def compute_target_pose(self, prev_ee_pose_at_base, action):
        # Keep the current rotation and change the position
        target_pose = sapien.Pose(action)
        arm_base_pose = self.arm_base_link.pose
        robot_base_pose = self.articulation.pose
        target_pose = robot_base_pose.inv() * arm_base_pose * target_pose
        target_pose.q = prev_ee_pose_at_base.q
        return target_pose


@dataclass
class PDEEPosMobileControllerConfig(ControllerConfig):
    lower: Union[float, Sequence[float]]
    upper: Union[float, Sequence[float]]
    stiffness: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    ee_link: str = None
    arm_base_link: str = None
    use_target: bool = False
    interpolate: bool = False
    controller_cls = PDEEPosMobileController


class PDEEPoseMobileController(PDEEPoseController):
    config: "PDEEPoseMobileControllerConfig"

    def _initialize_joints(self):
        super()._initialize_joints()
        if self.config.arm_base_link:
            self.arm_base_link = get_entity_by_name(
                self.articulation.get_links(), self.config.arm_base_link
            )
        else:
            # The child link of last joint is assumed to be the end-effector.
            self.arm_base_link = self.joints[0].get_parent_link()
        self.arm_base_link_idx = self.articulation.get_links().index(self.arm_base_link)

    def compute_target_pose(self, _, action):
        target_pos, target_rot = action[0:3], action[3:6]
        target_quat = Rotation.from_rotvec(target_rot).as_quat()[[3, 0, 1, 2]]
        target_pose = sapien.Pose(target_pos, target_quat)

        arm_base_pose = self.arm_base_link.pose
        robot_base_pose = self.articulation.pose
        return robot_base_pose.inv() * arm_base_pose * target_pose


@dataclass
class PDEEPoseMobileControllerConfig(ControllerConfig):
    pos_lower: Union[float, Sequence[float]]
    pos_upper: Union[float, Sequence[float]]
    rot_bound: float
    stiffness: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    ee_link: str = None
    arm_base_link: str = None
    use_target: bool = False
    interpolate: bool = False
    controller_cls = PDEEPoseMobileController
