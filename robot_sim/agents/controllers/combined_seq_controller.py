import numpy as np
import sapien.core as sapien

from mani_skill2.agents.base_controller import DictController, flatten_action_spaces
from robot_sim.agents.controllers import PDFollowJointController


class CombinedSequenceController(DictController):
    def __init__(self, config, articulation: sapien.Articulation, control_freq: int, sim_freq: int = None,
                 balance_passive_force=True):
        super().__init__(config.configs, articulation, control_freq, sim_freq, balance_passive_force)

    def _preprocess_action(self, action: np.ndarray):
        action_dim = self.action_space.shape[0]
        assert action.shape == (action.shape[0], action_dim)
        if self._normalize_action:
            action = self._clip_and_scale_action(action)
        return action

    def _initialize_action_space(self):
        super()._initialize_action_space()
        self.action_space, self.action_mapping = flatten_action_spaces(
            self.action_space.spaces
        )

    def set_action(self, action: np.ndarray):
        # Sanity check
        action_dim = self.action_space.shape[0]
        assert action.shape == (action.shape[0], action_dim)

        for uid, controller in self.controllers.items():
            start, end = self.action_mapping[uid]
            if isinstance(controller, PDFollowJointController):
                controller.set_action(action[:, start:end])
            else:
                controller.set_action(action[0, start:end])

    def to_action_dict(self, action: np.ndarray):
        """Convert a flat action to a dict of actions."""
        # Sanity check
        action_dim = self.action_space.shape[0]
        assert action.shape == (action.shape[0], action_dim)

        action_dict = {}
        for uid, controller in self.controllers.items():
            start, end = self.action_mapping[uid]
            if isinstance(controller, PDFollowJointController):
                action_dict[uid] = action[:, start:end]
            else:
                action_dict[uid] = action[0, start:end]
        return action_dict

    def from_action_dict(self, action_dict: dict):
        """Convert a dict of actions to a flat action."""
        seq_len = 1
        for k, v in action_dict.items():
            if len(v.shape) == 1:
                action_dict[k] = v[np.newaxis]
            else:
                assert len(v.shape) == 2
                seq_len = max(seq_len, v.shape[0])
        for k, v in action_dict.items():
            if v.shape[0] < seq_len:
                pad = v[-1, np.newaxis].repeat(seq_len - v.shape[0], axis=0)
                action_dict[k] = np.concatenate([v, pad], axis=0)
        return np.hstack([action_dict[uid] for uid in self.controllers])


class CombinedSequenceControllerConfig():
    configs = {}
    controller_cls = CombinedSequenceController
