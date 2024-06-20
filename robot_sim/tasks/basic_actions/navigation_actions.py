from typing import List
import numpy as np
from sapien.core import Pose
from transforms3d.euler import euler2quat, quat2euler
import logging

from robot_sim.tasks.basic_actions import BasicAction
from robot_sim.tasks.basic_actions.trajectory_actions import FollowTrajectoryAction
from robot_sim.envs.basic_env import TaskBaseEnv


class NavigateAction(BasicAction):
    def __init__(self, target_pos: List, target_yaw: float, env: TaskBaseEnv, frame='world', attached_obj_list: List = [],
                 max_steps: int = 100):
        super().__init__(env, max_steps)
        self.target_pose = Pose([target_pos[0], target_pos[1], 0], euler2quat(0, 0, target_yaw, 'sxyz'))
        if frame == 'world':
            self.target_pose = self.agent.robot.pose.inv() * self.target_pose
        else:
            assert frame == 'root'
        self.attached_obj_list = attached_obj_list
        self.target_qpos = np.array([self.target_pose.p[0], self.target_pose.p[1], quat2euler(self.target_pose.q, 'sxyz')[2]])

    def act(self):
        cur_point_cloud = self.env.get_point_cloud(exclude_objs=self.attached_obj_list)
        attached_obj_info = [{'mesh_file': obj.config.collision_path, 'pose_in_world': obj.pose * obj.origin_offset}
                             for obj in self.attached_obj_list]
        plan = self.agent.path_planning(self.target_pose, point_cloud=cur_point_cloud, attatched_objs=attached_obj_info,
                                        group='base', frame='root')
        if plan['status'] == 'Success' and (len(plan) <= self.max_steps):
            follow_traj_action = FollowTrajectoryAction(plan, self.agent.robot.get_active_joints()[-1].get_drive_target(),
                                                        self.env, self.max_steps)
            succ, self.datalog = follow_traj_action.run()
            return succ
        elif plan['status'] != 'Success':
            logging.debug(plan['status'])
        elif len(plan) > self.max_steps:
            logging.debug('current steps (%d) is longer than max steps (%d)' % (len(plan), self.max_steps))
        return False

    def check_success(self):
        position_thresh = self.check_success_input.get('position_thresh', 0.02)
        rotation_thresh = self.check_success_input.get('rotation_thresh', np.deg2rad(5))
        cur_qpos = self.agent.robot.get_qpos()
        return (np.linalg.norm(cur_qpos[:2] - self.target_qpos[:2]) <= position_thresh
                and np.abs(cur_qpos[2] - self.target_qpos[2]) <= rotation_thresh)

    def get_goal_description(self):
        return ('Move base to [%f, %f] with rotation %f under robot root frame' %
                (self.target_qpos[0], self.target_qpos[1], self.target_qpos[2]))
