import numpy as np
from sapien.core import Pose
from copy import deepcopy
from scipy.spatial.transform import Rotation as R

from robot_sim.tasks.control_tasks import BaseControlTask
from robot_sim.tasks.basic_actions.gripper_actions import GripperOpenCloseAction, GraspAction
# from robot_sim.tasks.basic_actions.end_effector_action import RelativeEEPoseAction, AbsoluteEEPoseAction
from robot_sim.tasks.basic_actions.trajectory_actions import AbsoluteLinearEEPoseAction, RelativeLinearEEPoseAction


class TableEnvControlTask(BaseControlTask):
    def __init__(self, *args, **kwargs):
        self.use_abs_pos = kwargs.pop('use_abs_pos', False)
        super().__init__(*args, **kwargs)
        self.target_obj = np.random.choice(self.env.objs)
        self.orig_obj_pos = deepcopy(self.target_obj.pose)
        self.grasp_action = GraspAction(self.target_obj, self.env)
        self.basket = self.env.get_articulation_by_name('basket')

    def reset(self):
        obs = super().reset()
        self.target_obj = np.random.choice(self.env.objs)
        self.orig_obj_pos = deepcopy(self.target_obj.pose)
        return obs

    def step(self, delta_ee_pos, delta_ee_rot, gripper_cmd):
        _, step_log = GripperOpenCloseAction('open' if gripper_cmd[0] > 0.5 else 'close', self.env).run()
        if self.use_abs_pos:
            _, obs_after_step = AbsoluteLinearEEPoseAction(
                Pose(delta_ee_pos, R.from_euler('xyz', delta_ee_rot, False).as_quat()[[3, 0, 1, 2]]),
                self.env, max_steps=2).run()
        else:
            _, obs_after_step = RelativeLinearEEPoseAction(
                Pose(delta_ee_pos, R.from_euler('xyz', delta_ee_rot, False).as_quat()[[3, 0, 1, 2]]),
                self.env, max_steps=2).run()
        step_log.extend(obs_after_step)
        self.cmd_log.append((delta_ee_pos, delta_ee_rot, gripper_cmd))
        self.exec_log.append(step_log)
        if self.env.render_mode == 'human':
            self.env.render()
        return obs_after_step.last_log

    def check_task_succeed(self, stage_idx):
        episode_succ = [False, False]
        # grasping phase
        if stage_idx == 1:
            # graped object?
            if self.env.agent.robot.get_active_joints()[-1].get_drive_target() < 0.02 and self.grasp_action.check_grasped():
                episode_succ[0] = True
            # object leaves the table?
            if episode_succ[0] and np.linalg.norm(self.orig_obj_pos.p - self.target_obj.pose.p) > 0.04:
                episode_succ[1] = True
        # dropping phase
        elif stage_idx == 2:
            # reach drop position?
            if np.linalg.norm(self.basket.pose.p[:2] - self.target_obj.pose.p[:2]) < 0.1:
                episode_succ[0] = True
            # gripper opened?
            if self.env.agent.robot.get_active_joints()[-1].get_drive_target() > 0.02:
                episode_succ[1] = True
        else:
            raise NotImplementedError
        return episode_succ
