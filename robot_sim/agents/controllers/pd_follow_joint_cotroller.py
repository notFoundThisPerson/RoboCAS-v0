import numpy as np
from dataclasses import dataclass
from typing import Union, Sequence

from mani_skill2.agents.controllers import PDJointPosVelControllerConfig, PDJointPosVelController


class PDFollowJointController(PDJointPosVelController):
    config: "PDFollowJointControllerConfig"

    def __init__(self, *args, **kwargs):
        if kwargs.get('config', args[0]).normalize_action:
            print('[Warning] Currently the "normalize_action" option is not supported, this option will be ignored')
            kwargs['config'].normalize_action = False
        super().__init__(*args, **kwargs)

    def _preprocess_action(self, action: np.ndarray):
        action_dim = self.action_space.shape[0]
        assert action.shape == (action.shape[0], action_dim)
        if self._normalize_action:
            action = self._clip_and_scale_action(action)
        return action

    def reset(self):
        super().reset()
        if self.config.use_delta:
            self._action = np.zeros_like(self._start_qpos)[np.newaxis]
        else:
            self._action = self._start_qpos[np.newaxis]
        vel_action = np.zeros_like(self._action)
        self._action = np.concatenate([self._action, vel_action], 1)

    def before_simulation_step(self):
        nq = self.action_space.shape[0] // 2
        if self._step < self._action.shape[0]:
            if self.config.use_delta:
                self.set_drive_targets(self.qpos + self._action[self._step, :nq])
            else:
                self.set_drive_targets(self._action[self._step, :nq])
            self.set_drive_velocity_targets(self._action[self._step, nq:])
        self._step += 1

    def set_action(self, action: np.ndarray):
        assert len(action.shape) == 2
        action = self._preprocess_action(action)
        nq = action.shape[1] // 2

        self._step = 0
        self._start_qpos = self.qpos
        self._action = action.copy()

        if self.config.use_delta:
            if self.config.use_target:
                self._target_qpos = self._target_qpos + np.sum(action[:, :nq], 0)
            else:
                self._target_qpos = self._start_qpos + np.sum(action[:, :nq], 0)
        else:
            self._target_qpos = action[-1, :nq]


@dataclass
class PDFollowJointControllerConfig(PDJointPosVelControllerConfig):
    vel_lower: Union[float, Sequence[float]] = -3.14
    vel_upper: Union[float, Sequence[float]] = 3.14
    normalize_action: bool = False
    controller_cls = PDFollowJointController
