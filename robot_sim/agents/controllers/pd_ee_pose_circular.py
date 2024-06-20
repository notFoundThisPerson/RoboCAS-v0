import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
from sapien.core import Pose
from gymnasium import spaces

from robot_sim.agents.controllers import PDEEPoseLinearControllerConfig, PDEEPoseLinearController


class PDEEPoseCircularController(PDEEPoseLinearController):
    config: "PDEEPoseCircularControllerConfig"

    def _initialize_action_space(self):
        # [axis_xyz, axis_rot, angle]
        self.action_space = spaces.Box(np.zeros([7]), np.zeros([7]), dtype=np.float32)

    def _preprocess_action(self, action: np.ndarray):
        action_dim = self.action_space.shape[0]
        assert action.shape == (action_dim,), (action.shape, action_dim)
        axis_pose = Pose(p=action[:3], q=R.from_rotvec(action[3:6]).as_quat()[[3, 0, 1, 2]])
        if self.config.frame == "base":
            pass
        elif self.config.frame == 'world':
            axis_pose = self.articulation.pose.inv() * axis_pose
        elif self.config.frame == 'ee':
            axis_pose = self.ee_pose_at_base * axis_pose
        else:
            raise NotImplementedError
        return axis_pose, action[6]

    def set_action(self, action: np.ndarray):
        self._step = 0
        self._start_qpos = self.qpos

        axis_pose_wrt_base, delta_angle = self._preprocess_action(action)
        self._action = self.path_interpolate(axis_pose_wrt_base, delta_angle)

        nq = len(self.joints)
        self._target_qpos = self._action[-1, :nq]

    def path_interpolate(self, axis_pose_wrt_base, delta_angle):
        if self.config.use_target:
            prev_ee_pose_at_base = self._target_pose
        else:
            prev_ee_pose_at_base = self.ee_pose_at_base

        ee_pose_wrt_joint = axis_pose_wrt_base.inv() * prev_ee_pose_at_base
        delta_pose_wrt_joint = Pose(q=R.from_euler('xyz', [delta_angle, 0, 0], False).as_quat()[[3, 0, 1, 2]])
        self._target_pose = axis_pose_wrt_base * delta_pose_wrt_joint * ee_pose_wrt_joint

        rotate_radius = np.linalg.norm(ee_pose_wrt_joint.p[1:])
        circular_dist = abs(delta_angle * rotate_radius)
        sim_dt = self.articulation.get_builder().get_scene().get_timestep()
        dist_limit_per_step = self.config.linear_vel_thresh * sim_dt
        seq_len = int(np.ceil(circular_dist / dist_limit_per_step))

        delta_pose_per_step = Pose(q=R.from_euler('xyz', [delta_angle / seq_len, 0, 0], False).as_quat()[[3, 0, 1, 2]])
        qpos_traj = []
        last_qpos = self.articulation.get_qpos()
        last_joint_pose = axis_pose_wrt_base
        for _ in range(seq_len):
            next_joint_pose = last_joint_pose * delta_pose_per_step
            next_pose = next_joint_pose * ee_pose_wrt_joint
            ik = self.compute_ik(next_pose, last_qpos)
            last_joint_pose = next_joint_pose
            if ik is None:
                continue
            qpos_traj.append(ik)
            last_qpos[self.joint_indices] = ik
        qpos_traj = np.stack(qpos_traj, 0)
        qvel_traj = np.zeros_like(qpos_traj)
        qvel_traj[1:-1] = (qpos_traj[2:] - qpos_traj[:-2]) / (sim_dt * 2)
        qvel_traj[0] = (qpos_traj[1] - self._start_qpos) / (sim_dt * 2)
        return np.concatenate([qpos_traj, qvel_traj], 1)


@dataclass
class PDEEPoseCircularControllerConfig(PDEEPoseLinearControllerConfig):
    controller_cls = PDEEPoseCircularController
