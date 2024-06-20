from sapien.core import Pose

from mani_skill2.agents.base_agent import BaseAgent
from robot_sim.utils import format_path


class MtBaseAgent(BaseAgent):
    def _load_articulation(self):
        self.urdf_path = format_path(self.urdf_path)
        super()._load_articulation()
        self.set_root_pose()

    def set_root_pose(self, pose=Pose()):
        self.robot.set_pose(pose)

    def reset(self, init_qpos=None):
        super().reset(init_qpos)
        for (joint, qpos) in zip(self.robot.get_active_joints(), self.robot.get_qpos()):
            joint.set_drive_target(qpos)
