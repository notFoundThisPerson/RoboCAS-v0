from scipy.spatial.transform import Rotation as R
from typing import Union
import sapien.core as sapien
import numpy as np

from robot_sim.tasks.basic_actions import BasicAction
from robot_sim.envs.basic_env import TaskBaseEnv
from robot_sim.utils import get_pose_error


class RelativeEEPoseAction(BasicAction):
    def __init__(self, rel_pose: Union[sapien.Pose, np.ndarray], env: TaskBaseEnv, frame: str = 'ee',
                 max_steps=50):
        super().__init__(env, max_steps)
        if isinstance(rel_pose, np.ndarray):
            rel_pose = sapien.Pose().from_transformation_matrix(rel_pose)
        if frame == 'ee':
            self.target_pose_world = self.agent.ee_pose_world * rel_pose
            self.rel_pose = rel_pose
        elif frame == 'world':
            self.target_pose_world = rel_pose * self.agent.ee_pose_world
            self.rel_pose = self.agent.ee_pose_world.inv() * self.target_pose_world
        elif frame == 'base':
            ee_pose_base = self.agent.base_pose_world.inv() * self.agent.ee_pose_world
            self.target_pose_world = self.agent.base_pose_world * rel_pose * ee_pose_base
            self.rel_pose = self.agent.ee_pose_world.inv() * self.target_pose_world
        else:
            assert frame == 'root'
            self.target_pose_world = self.agent.robot.pose * rel_pose * self.agent.ee_pose_root
            self.rel_pose = self.agent.ee_pose_world.inv() * self.target_pose_world
        self.target_ctrl_mode = 'base_pd_delta_pos_arm_pd_ee_delta_pose'

    def act(self):
        action_dict = {
            'base': [0, 0, 0],
            'arm': [*self.rel_pose.p, *R.from_quat(self.rel_pose.q[[1, 2, 3, 0]]).as_rotvec()],
            'gripper': self.agent.robot.get_active_joints()[-1].get_drive_target()
        }
        action = self.agent.controller.from_action_dict(action_dict)
        return super().act(action)

    def check_success(self):
        pos_err_thresh = self.check_success_input.get('pos_err_thresh', 2e-3)
        orn_err_thresh = self.check_success_input.get('orn_err_thresh', np.deg2rad(5))
        pos_err, orn_err = get_pose_error(self.agent.ee_pose_world, self.target_pose_world)
        return pos_err <= pos_err_thresh and orn_err <= orn_err_thresh

    def get_goal_description(self):
        return ('move gripper relative: xyz=[%.3f, %.3f, %.3f], orn=[%.3f, %.3f, %.3f, %.3f]' %
                (*self.rel_pose.p, *self.rel_pose.q[[1, 2, 3, 0]]))


class AbsoluteEEPoseAction(BasicAction):
    def __init__(self, target_pose: Union[sapien.Pose, np.ndarray], env: TaskBaseEnv, frame: str = 'base',
                 max_steps=50):
        super().__init__(env, max_steps)
        self.frame = frame
        if isinstance(target_pose, np.ndarray):
            target_pose = sapien.Pose().from_transformation_matrix(target_pose)
        if self.frame == 'base':
            self.target_pose_root = self.agent.robot.pose.inv() * self.agent.arm_base_link.pose * target_pose
        elif self.frame == 'root':
            self.target_pose_root = target_pose
        elif self.frame == 'world':
            self.target_pose_root = self.agent.robot.pose.inv() * target_pose
        else:
            assert frame == 'ee'
            self.target_pose_root = self.agent.ee_pose_root * target_pose
        self.target_ctrl_mode = 'base_pd_delta_pos_arm_pd_ee_abs_pose_wrt_robot_base'

    def act(self):
        action_dict = {
            'base': [0, 0, 0],
            'arm': [*self.target_pose_root.p, *R.from_quat(self.target_pose_root.q[[1, 2, 3, 0]]).as_rotvec()],
            'gripper': self.agent.robot.get_active_joints()[-1].get_drive_target()
        }
        action = self.agent.controller.from_action_dict(action_dict)
        return super().act(action)

    def check_success(self):
        pos_err_thresh = self.check_success_input.get('pos_err_thresh', 2e-3)
        orn_err_thresh = self.check_success_input.get('orn_err_thresh', np.deg2rad(5))
        pos_err, orn_err = get_pose_error(self.agent.ee_pose_root, self.target_pose_root)
        return pos_err <= pos_err_thresh and orn_err <= orn_err_thresh

    def get_goal_description(self):
        return ('move the gripper to target pose w.r.t. root frame: xyz=[%.3f, %.3f, %.3f], orn=[%.3f, %.3f, %.3f, %.3f]' %
                (*self.target_pose_root.p, *self.target_pose_root.q[[1, 2, 3, 0]]))
