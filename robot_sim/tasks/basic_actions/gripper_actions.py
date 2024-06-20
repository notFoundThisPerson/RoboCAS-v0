import numpy as np
from typing import Union
import sapien.core as sapien
import logging

from robot_sim.envs.basic_env import TaskBaseEnv
from robot_sim.envs.objects.env_object import BaseObject
from robot_sim.tasks.basic_actions import BasicAction


class GripperAction(BasicAction):
    def __init__(self, target_pos: float, env: TaskBaseEnv, max_steps: int = 5):
        super().__init__(env, max_steps)
        self.target_pos = target_pos
        self.target_ctrl_mode = 'base_pd_joint_vel_arm_pd_joint_vel'

    def act(self):
        if abs(self.agent.robot.get_qpos()[-1] - self.target_pos) < 1e-3:
            return True
        base_vel = [0] * len(self.agent.config.base_joint_names)
        arm_vel = [0] * len(self.agent.config.arm_joint_names)
        action = np.array(base_vel + arm_vel + [self.target_pos])
        ret = super().act(action)
        if not ret:
            logging.debug('Failed to move gripper to %f in %d steps' % (self.target_pos, self.max_steps))
        else:
            self.env.step(action)
        return ret

    def check_success(self):
        pos_thresh = self.check_success_input.get('pos_thresh', 2e-3)
        vel_thresh = self.check_success_input.get('vel_thresh', 1e-3)
        pos_error = abs(self.agent.robot.get_qpos()[-1] - self.target_pos)
        vel_error = abs(self.agent.robot.get_qvel()[-1])
        pos_satisfied = pos_error < pos_thresh
        vel_satisfied = vel_error < vel_thresh
        logging.debug('pos_error: %f (%s), vel_error: %f (%s)' % (pos_error, pos_satisfied, vel_error,  vel_satisfied))
        return pos_satisfied or vel_satisfied

    def get_goal_description(self):
        return 'Move the gripper to %.3f' % self.target_pos


class GraspAction(GripperAction):
    def __init__(self, target_obj: Union[BaseObject, sapien.Articulation, sapien.Actor], env: TaskBaseEnv, max_steps: int = 5):
        super().__init__(env.agent.gripper_close_width, env, max_steps)
        self.target_obj = target_obj

    def check_success(self):
        return super().check_success() and self.check_grasped()

    def check_grasped(self):
        finger1 = [self.agent.finger1_link, self.agent.fingertip1_link]
        finger2 = [self.agent.finger2_link, self.agent.fingertip2_link]
        if isinstance(self.target_obj, BaseObject):
            target_link = self.target_obj.model
        else:
            target_link = self.target_obj
        if isinstance(target_link, sapien.Articulation):
            target_link = target_link.get_links()
        else:
            target_link = [target_link]
        finger1_cnt, finger2_cnt = False, False
        for cnt in self.env.scene.get_contacts():
            min_separation = min([p.separation for p in cnt.points])
            if min_separation > 2e-3:
                continue
            if cnt.actor0 in target_link:
                if cnt.actor1 in finger1:
                    finger1_cnt = True
                    logging.debug('actor0: %s, actor1: %s' % (cnt.actor0.name, cnt.actor1.name))
                elif cnt.actor1 in finger2:
                    finger2_cnt = True
                    logging.debug('actor0: %s, actor1: %s' % (cnt.actor0.name, cnt.actor1.name))
            elif cnt.actor1 in target_link:
                if cnt.actor0 in finger1:
                    finger1_cnt = True
                    logging.debug('actor0: %s, actor1: %s' % (cnt.actor0.name, cnt.actor1.name))
                elif cnt.actor0 in finger2:
                    finger2_cnt = True
                    logging.debug('actor0: %s, actor1: %s' % (cnt.actor0.name, cnt.actor1.name))
            if finger1_cnt and finger2_cnt:
                return True
        return False

    def get_goal_description(self):
        return 'Grasp the %s' % self.target_obj.name if isinstance(self.target_obj, BaseObject) else 'target'


class GripperOpenCloseAction(GripperAction):
    def __init__(self, target_status: str, env: TaskBaseEnv, max_steps: int = 5):
        assert target_status in ['open', 'close']
        self.target_status = target_status
        joint_limits = env.agent.robot.get_active_joints()[-1].get_limits()[0]
        target_pos = joint_limits[0] if target_status == 'close' else joint_limits[1]
        super().__init__(target_pos, env, max_steps)

    def get_goal_description(self):
        return '%s the gripper' % self.target_status
