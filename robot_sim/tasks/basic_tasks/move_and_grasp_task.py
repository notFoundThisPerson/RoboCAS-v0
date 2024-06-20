import numpy as np
from sapien.core import Pose
from typing import Union
import logging

from robot_sim.tasks.basic_actions import BasicAction
from robot_sim.envs.objects.env_object import BaseObject
from robot_sim.tasks.basic_actions.trajectory_actions import MoveToPoseAction, RelativeLinearEEPoseAction
# from robot_sim.tasks.basic_actions.navigation_actions import NavigateAction
from robot_sim.tasks.basic_actions.gripper_actions import GraspAction
from robot_sim.envs.basic_env import TaskBaseEnv
from robot_sim.utils import DataLog


class MoveAndGraspTask(BasicAction):
    def __init__(self, target_obj: BaseObject, grasp_pose: Union[Pose, np.ndarray], env: TaskBaseEnv,
                 pose_frame='world', move_group='arm',
                 pre_grasp_offset_ee=Pose([0, 0, -0.1]), after_grasp_offset_base=Pose([0, 0, 0.05]),
                 max_steps=100):
        super().__init__(env, max_steps)

        self.pre_grasp_offset = pre_grasp_offset_ee
        self.after_grasp_offset = after_grasp_offset_base
        if isinstance(grasp_pose, np.ndarray):
            grasp_pose = Pose.from_transformation_matrix(grasp_pose)
        self.grasp_pose = grasp_pose
        self.pose_frame = pose_frame
        self.move_group = move_group
        self.grasp_action = GraspAction(target_obj, self.env, max_steps=10)
        self.grasp_action.check_success_input['success_fn'] = self.grasp_action.check_grasped

    def act(self):
        pre_grasp_pose = self.grasp_pose * self.pre_grasp_offset
        succ, obs = MoveToPoseAction(pre_grasp_pose, 'open', self.env, self.pose_frame, max_steps=self.max_steps,
                                     move_group=self.move_group).run()
        if succ:
            self.datalog.extend(obs)
            _, obs = RelativeLinearEEPoseAction(self.pre_grasp_offset.inv() * Pose([0, 0, 0.01]), self.env, 'ee').run()
        else:
            succ, obs = MoveToPoseAction(self.grasp_pose, 'open', self.env, self.pose_frame,
                                         max_steps=self.max_steps, move_group=self.move_group).run()
        if not succ:
            logging.debug('Failed to move and grasp target %s: move failed' % self.grasp_action.target_obj.name)
            return False
        self.datalog.extend(obs)
        succ, obs = self.grasp_action.run()
        if not succ:
            logging.debug('Failed to move and grasp target %s: grasp failed' % self.grasp_action.target_obj.name)
            return False
        for _ in range(int(0.1 * self.env.sim_freq)):
            self.env.scene.step()
        self.datalog.extend(obs)
        _, obs = RelativeLinearEEPoseAction(self.after_grasp_offset, self.env, 'base').run()
        self.datalog.extend(obs)
        if not self.success():
            return False
        return True

    def check_success(self):
        self.check_success_input.update(self.check_success_input)
        return self.grasp_action.check_grasped()

    def get_goal_description(self):
        return 'Move and grasp the %s' % self.grasp_action.target_obj.name
