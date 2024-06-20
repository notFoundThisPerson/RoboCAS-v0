import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
from sapien.core import Pose

from mani_skill2.agents.controllers import PDEEPoseControllerConfig, PDEEPoseController


class PDEEPoseLinearController(PDEEPoseController):
    config: "PDEEPoseLinearControllerConfig"

    def __init__(self, *args, **kwargs):
        if kwargs.get('config', args[0]).normalize_action:
            print('[Warning] Currently the "normalize_action" option is not supported, this option will be ignored')
            kwargs['config'].normalize_action = False
        super().__init__(*args, **kwargs)

    def _clip_and_scale_action(self, action):
        raise NotImplementedError

    def reset(self):
        super().reset()
        if self.config.use_delta:
            self._action = np.zeros_like(self._start_qpos)[np.newaxis]
        else:
            self._action = self._start_qpos[np.newaxis]
        vel_action = np.zeros_like(self._action)
        self._action = np.concatenate([self._action, vel_action], 1)

    def before_simulation_step(self):
        nq = len(self.joints)
        if self._step < self._action.shape[0]:
            self.set_drive_targets(self._action[self._step, :nq])
            self.set_drive_velocity_targets(self._action[self._step, nq:])
        self._step += 1

    def set_action(self, action: np.ndarray):
        action = self._preprocess_action(action)

        self._step = 0
        self._start_qpos = self.qpos

        if self.config.use_target:
            prev_ee_pose_at_base = self._target_pose
        else:
            prev_ee_pose_at_base = self.ee_pose_at_base

        self._target_pose = self.compute_target_pose(prev_ee_pose_at_base, action)
        self._action = self.path_interpolate(prev_ee_pose_at_base, self._target_pose)

        nq = len(self.joints)
        self._target_qpos = self._action[-1, :nq]

    def path_interpolate(self, start, end):
        sim_dt = self.articulation.get_builder().get_scene().get_timestep()
        dist_limit_per_step = self.config.linear_vel_thresh * sim_dt
        seq_length = max(int(np.ceil(np.linalg.norm(self._target_pose.p - self.ee_pose_at_base.p) / dist_limit_per_step)), 1)

        start_rotvec = R.from_quat(start.q[[1, 2, 3, 0]]).as_rotvec()
        end_rotvec = R.from_quat(end.q[[1, 2, 3, 0]]).as_rotvec()
        pos_step = (end.p - start.p) / seq_length
        orn_step = (end_rotvec - start_rotvec) / seq_length

        step_idx = np.arange(1, seq_length + 1)[:, np.newaxis]
        pos_list = start.p[np.newaxis] + step_idx * pos_step[np.newaxis]
        orn_list = start_rotvec[np.newaxis] + step_idx * orn_step[np.newaxis]
        orn_list = R.from_rotvec(orn_list).as_quat()[:, [3, 0, 1, 2]]

        qpos_traj = []
        last_qpos = self.articulation.get_qpos()
        for pos, orn in zip(pos_list, orn_list):
            next_pose = Pose(pos, orn)
            ik = self.compute_ik(next_pose, last_qpos)
            if ik is None:
                continue
            qpos_traj.append(ik)
            last_qpos[self.joint_indices] = ik
        if len(qpos_traj) == 0:
            qpos_traj = [self.articulation.get_qpos()[self.joint_indices]]
        qpos_traj = np.stack(qpos_traj, 0)
        qvel_traj = np.zeros_like(qpos_traj)
        if len(qpos_traj) > 1:
            qvel_traj[1:-1] = (qpos_traj[2:] - qpos_traj[:-2]) / (sim_dt * 2)
            qvel_traj[0] = (qpos_traj[1] - self._start_qpos) / (sim_dt * 2)
        return np.concatenate([qpos_traj, qvel_traj], 1)

    def compute_ik(self, target_pose, start_qpos, max_iterations=100):
        # Assume the target pose is defined in the base frame
        result, success, error = self.pmodel.compute_inverse_kinematics(
            self.ee_link_idx,
            target_pose,
            initial_qpos=start_qpos,
            active_qmask=self.qmask,
            max_iterations=max_iterations,
        )
        if success:
            return result[self.joint_indices]
        else:
            return None

    def set_drive_velocity_targets(self, targets):
        for i, joint in enumerate(self.joints):
            joint.set_drive_velocity_target(targets[i])


@dataclass
class PDEEPoseLinearControllerConfig(PDEEPoseControllerConfig):
    linear_vel_thresh: float = 0.5
    normalize_action: bool = False
    controller_cls = PDEEPoseLinearController
