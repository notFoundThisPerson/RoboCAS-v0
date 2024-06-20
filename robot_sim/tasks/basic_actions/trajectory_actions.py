import numpy as np
from typing import Union, List
import logging
from sapien import core as sapien
from sapien.core import Pose
from scipy.spatial.transform import Rotation as R

from robot_sim.envs.basic_env import TaskBaseEnv
from robot_sim.envs.objects.graspnet_object import GraspNetObject
from robot_sim.tasks.basic_actions import BasicAction
from robot_sim.utils import get_pose_error


class FollowTrajectoryAction(BasicAction):
    def __init__(self, path: dict, gripper_open_width: float, env: TaskBaseEnv, max_steps: int = 50):
        super().__init__(env, max_steps)
        self.target_ctrl_mode = 'base_follow_joint_arm_follow_joint'
        self.action_dict = {
            'base': np.concatenate([path['position'][:, :3], path['velocity'][:, :3]], axis=1),
            'arm': np.concatenate([path['position'][:, 3:], path['velocity'][:, 3:]], axis=1),
            'gripper': np.array([gripper_open_width])
        }

    def act(self):
        path = self.agent.controller.from_action_dict(self.action_dict)
        path_obs = []
        for step_start in range(0, path.shape[0], self.env.sim_steps_per_control):
            step_end = step_start + self.env.sim_steps_per_control
            obs = self.env.step(path[step_start:step_end])[0]
            if self.env.render_mode == 'human':
                self.env.render_human()
            path_obs.append(obs)
        self.datalog = self.datalog.from_list(path_obs)
        return True

    def check_success(self):
        joint_err_thresh = self.check_success_input.get('joint_err_thresh', np.deg2rad(2))
        cur_qpos = self.agent.robot.get_qpos()
        arm_dof = self.action_dict['arm'].shape[1] // 2
        return (np.all(np.abs(self.action_dict['base'][-1][:3] - cur_qpos[:3]) < joint_err_thresh) and
                np.all(np.abs(self.action_dict['arm'][-1][:arm_dof] - cur_qpos[3:arm_dof + 3]) < joint_err_thresh))

    def get_goal_description(self):
        return 'Follow trajectory, trajectory length=%d' % len(self.action_dict['arm'])


class MoveToPoseAction(BasicAction):
    def __init__(self, target_pose: Union[Pose, np.ndarray], gripper_status: str, env: TaskBaseEnv, pose_frame='world',
                 max_steps: int = 100, attached_obj_list: List[GraspNetObject] = [], move_group='arm'):
        super().__init__(env, max_steps)
        if isinstance(target_pose, Pose):
            target_pose = target_pose.to_transformation_matrix()
        self.target_pose = target_pose
        assert gripper_status in ['open', 'close']
        self.gripper_open_width = self.agent.gripper_close_width if gripper_status == 'close' else self.agent.gripper_open_width
        self.attached_obj_list = attached_obj_list
        self.move_group = move_group
        self.pose_frame = pose_frame

    def act(self):
        cur_point_cloud = self.env.get_point_cloud(exclude_objs=self.attached_obj_list)
        attached_obj_info = [{'mesh_file': obj.config.collision_path, 'pose_in_world': obj.pose * obj.origin_offset}
                             for obj in self.attached_obj_list]
        plan = self.agent.path_planning(self.target_pose, point_cloud=cur_point_cloud, attatched_objs=attached_obj_info,
                                        group=self.move_group, frame=self.pose_frame)
        if plan['status'] == 'Success' and (len(plan) <= self.max_steps):
            follow_traj_action = FollowTrajectoryAction(plan, self.gripper_open_width, self.env, self.max_steps)
            succ, self.datalog = follow_traj_action.run()
            return succ
        elif plan['status'] != 'Success':
            logging.debug(plan['status'])
        elif len(plan) > self.max_steps:
            logging.debug('current steps (%d) is longer than max steps (%d)' % (len(plan), self.max_steps))
        return False

    def check_success(self):
        pos_threshold = self.check_success_input.get('pos_threshold', 2e-3)
        orn_threshold = self.check_success_input.get('orn_threshold', np.deg2rad(5))
        if self.pose_frame == 'world':
            cur_ee_pose = self.agent.ee_pose_world
        elif self.pose_frame == 'root':
            cur_ee_pose = self.agent.ee_pose_root
        elif self.pose_frame == 'base':
            cur_ee_pose = self.agent.base_pose_world.inv() * self.agent.ee_pose_world
        else:
            raise ValueError
        cur_ee_pose = cur_ee_pose.to_transformation_matrix()
        pos_err, orn_err = get_pose_error(cur_ee_pose, self.target_pose)
        return pos_err <= pos_threshold and orn_err <= orn_threshold

    def get_goal_description(self):
        return ('Move to position [%.3f, %.3f, %.3f] and orientation [%.3f, %.3f, %.3f, %.3f]' %
                (*self.target_pose[:3, 3], *R.from_matrix(self.target_pose[:3, :3]).as_quat()))


class AbsoluteLinearEEPoseAction(BasicAction):
    def __init__(self, target_pose: Union[sapien.Pose, np.ndarray], env: TaskBaseEnv, frame: str = 'root',
                 move_group: str = 'arm', max_steps=50):
        super().__init__(env, max_steps)
        self.frame = frame
        self.joint_vel_history = []
        if isinstance(target_pose, np.ndarray):
            target_pose = sapien.Pose().from_transformation_matrix(target_pose)
        if frame == 'root':
            self.target_pose_root = target_pose
        elif frame == 'world':
            self.target_pose_root = self.agent.robot.pose.inv() * target_pose
        elif frame == 'base':
            self.target_pose_root = self.agent.robot.pose.inv() * self.agent.arm_base_link.pose * target_pose
        else:
            assert frame == 'ee'
            self.target_pose_root = self.agent.ee_pose_root * target_pose
        self.move_group = move_group
        if move_group == 'arm':
            self.target_ctrl_mode = 'base_pd_delta_pos_arm_pd_ee_abs_pose_linear_wrt_robot_base'
        elif move_group == 'full':
            self.target_ctrl_mode = 'global_pd_ee_abs_pose_linear_wrt_robot_base'
        else:
            raise NotImplementedError(f'{move_group} not in ["arm", "full"]')

    def act(self):
        if self.move_group == 'arm':
            action_dict = {
                'base': [0, 0, 0],
                'arm': [*self.target_pose_root.p, *R.from_quat(self.target_pose_root.q[[1, 2, 3, 0]]).as_rotvec()],
                'gripper': self.agent.robot.get_active_joints()[-1].get_drive_target()
            }
        else:
            action_dict = {
                'global': [*self.target_pose_root.p, *R.from_quat(self.target_pose_root.q[[1, 2, 3, 0]]).as_rotvec()],
                'gripper': self.agent.robot.get_active_joints()[-1].get_drive_target()
            }
        action = self.agent.controller.from_action_dict(action_dict)
        return super().act(action)

    def check_success(self):
        pos_err_thresh = self.check_success_input.get('pos_err_thresh', 0.02)
        orn_err_thresh = self.check_success_input.get('orn_err_thresh', np.deg2rad(5))
        joint_vel_thresh = self.check_success_input.get('joint_vel_thresh', np.deg2rad(3))
        pos_err, orn_err = get_pose_error(self.agent.ee_pose_root, self.target_pose_root)
        self.joint_vel_history.append(max(abs(self.agent.robot.get_qvel())) < joint_vel_thresh)
        return ((pos_err <= pos_err_thresh and orn_err <= orn_err_thresh) or
                (len(self.joint_vel_history) > 5 and all(self.joint_vel_history[-5:])))

    def get_goal_description(self):
        return ('move gripper under root frame: xyz=[%.3f, %.3f, %.3f], orn=[%.3f, %.3f, %.3f, %.3f]' %
                (*self.target_pose_root.p, *self.target_pose_root.q[[1, 2, 3, 0]]))


class RelativeLinearEEPoseAction(AbsoluteLinearEEPoseAction):
    def __init__(self, rel_pose: Union[sapien.Pose, np.ndarray], env: TaskBaseEnv, frame: str = 'ee', move_group: str = 'arm',
                 max_steps=50):
        self.rel_pose = rel_pose
        self.rel_frame = frame
        if isinstance(rel_pose, np.ndarray):
            rel_pose = sapien.Pose().from_transformation_matrix(rel_pose)
        agent = env.agent
        if frame == 'ee':
            target_pose_root = agent.ee_pose_root * rel_pose
        elif frame == 'world':
            target_pose_root = agent.robot.pose.inv() * rel_pose * agent.ee_pose_world
        elif frame == 'base':
            ee_pose_base = agent.base_pose_world.inv() * agent.ee_pose_world
            target_pose_root = agent.robot.pose.inv() * agent.base_pose_world * rel_pose * ee_pose_base
        else:
            assert frame == 'root'
            target_pose_root = rel_pose * agent.ee_pose_root
        super().__init__(target_pose_root, env, 'root', move_group, max_steps)

    def get_goal_description(self):
        return ('move gripper relative under %s frame: xyz=[%.3f, %.3f, %.3f], orn=[%.3f, %.3f, %.3f, %.3f]' %
                (self.rel_frame, *self.rel_pose.p, *self.rel_pose.q[[1, 2, 3, 0]]))


class CircularEEPoseAction(BasicAction):
    def __init__(self, joint: sapien.Joint, delta_angle: float, env: TaskBaseEnv, move_group: str = 'arm', max_steps: int = 50):
        super().__init__(env, max_steps)
        self.target_joint = joint
        self.delta_angle = delta_angle
        joint_pose_world = self.target_joint.get_global_pose()
        delta_pose_joint = Pose(q=R.from_euler('xyz', [delta_angle, 0, 0], False).as_quat()[[3, 0, 1, 2]])
        self.target_pose_world = joint_pose_world * delta_pose_joint * joint_pose_world.inv() * self.agent.ee_pose_world
        self.move_group = move_group
        if move_group == 'arm':
            self.target_ctrl_mode = 'base_pd_delta_pos_arm_pd_ee_abs_pose_circular_wrt_world'
        elif move_group == 'full':
            self.target_ctrl_mode = 'global_pd_ee_abs_pose_circular_wrt_world'
        else:
            raise NotImplementedError(f'{move_group} not in ["arm", "full"]')

    def act(self):
        joint_pose = self.target_joint.get_global_pose()
        if self.move_group == 'arm':
            action_dict = {
                'base': [0, 0, 0],
                'arm': [*joint_pose.p, *R.from_quat(joint_pose.q[[1, 2, 3, 0]]).as_rotvec(), self.delta_angle],
                'gripper': self.agent.robot.get_active_joints()[-1].get_drive_target()
            }
        else:
            action_dict = {
                'global': [*joint_pose.p, *R.from_quat(joint_pose.q[[1, 2, 3, 0]]).as_rotvec(), self.delta_angle],
                'gripper': self.agent.robot.get_active_joints()[-1].get_drive_target()
            }
        action = self.agent.controller.from_action_dict(action_dict)
        step_obs = []
        for _ in range(self.max_steps):
            obs = self.env.step(action)[0]
            step_obs.append(obs)
            if self.env.render_mode == 'human':
                self.env.render_human()
            if self.success():
                self.datalog, self.datalog.from_list(step_obs)
                return True
            action = None
        return False

    def check_success(self):
        pos_err_thresh = self.check_success_input.get('pos_err_thresh', 0.01)
        orn_err_thresh = self.check_success_input.get('orn_err_thresh', np.deg2rad(5))
        pos_err, orn_err = get_pose_error(self.agent.ee_pose_world, self.target_pose_world)
        return pos_err <= pos_err_thresh and orn_err <= orn_err_thresh

    def get_goal_description(self):
        return 'move gripper circular wrt %s by %.3f rad' % (self.target_joint.name, self.delta_angle)
