from sapien.core import Pose
import logging

from mani_skill2.utils.sapien_utils import get_entity_by_name
from robot_sim.tasks.basic_actions import BasicAction
from robot_sim.envs.objects.articulation_object import ArticulationObject, OperationConfig
from robot_sim.tasks.basic_tasks.move_and_grasp_task import MoveAndGraspTask
from robot_sim.tasks.basic_actions.trajectory_actions import RelativeLinearEEPoseAction, CircularEEPoseAction
from robot_sim.tasks.basic_actions.gripper_actions import GripperOpenCloseAction
from robot_sim.envs.basic_env import TaskBaseEnv
from robot_sim.utils import DataLog, create_fixed_constraint


class OperationTaskBase(BasicAction):
    def __init__(self, target_obj: ArticulationObject, op_config: OperationConfig, target_status: str, env: TaskBaseEnv,
                 move_group='arm', max_steps=100):
        assert target_status in ['open', 'close']
        super().__init__(env, max_steps)
        self.target_obj = target_obj
        self.config = op_config
        self.target_status = target_status
        self.target_joint = get_entity_by_name(self.target_obj.model.get_active_joints(), self.config.joint_name)
        self.target_joint_idx = self.target_obj.model.get_active_joints().index(self.target_joint)
        self.target_link = get_entity_by_name(self.target_obj.model.get_links(), self.config.link_name)
        self.limit = self.target_joint.get_limits()[0]
        self.move_group = move_group
        if target_status == 'open':
            self.target_pos = self.limit[1]
        else:
            self.target_pos = self.limit[0]

    @property
    def cur_target_qpos(self):
        return self.target_obj.model.get_qpos()[self.target_joint_idx]

    def check_success(self):
        percent_thresh = self.check_success_input.get('percent_thresh', 0.2)
        qpos_thresh = percent_thresh * (self.limit[1] - self.limit[0])
        if self.target_status == 'open':
            return self.limit[1] - self.cur_target_qpos <= qpos_thresh
        else:
            return self.cur_target_qpos - self.limit[0] <= qpos_thresh

    def before_operation(self):
        grasp_pose = self.target_link.pose * self.config.handle_pose
        succ, obs = MoveAndGraspTask(self.target_link, grasp_pose, self.env, 'world', self.move_group).run()
        self.datalog.extend(obs)
        if not succ:
            logging.debug('Failed to %s the slide door of the %s: unable to reach the handle' %
                             (self.target_status, self.target_obj.name))
            return False
        return True

    def operation_step(self, **kwargs):
        raise NotImplementedError

    def after_operation(self):
        return True

    def act(self, **kwargs):
        if self.success():
            return True, self.datalog

        succ = self.before_operation()
        if not succ:
            return False
        succ = self.operation_step(**kwargs)
        if not succ:
            return False
        succ = self.after_operation()
        if not succ:
            return False
        return True


class SlideOperationTask(OperationTaskBase):
    def operation_step(self):
        if self.target_status == 'open':
            delta_dist = self.limit[1] - self.cur_target_qpos
        else:
            delta_dist = self.limit[0] - self.cur_target_qpos
        ee_pose_wrt_joint = self.target_joint.get_global_pose().inv() * self.agent.ee_pose_world
        delta_pose_wrt_joint = Pose(p=[delta_dist, 0, 0])
        delta_pose_wrt_ee = ee_pose_wrt_joint.inv() * delta_pose_wrt_joint * ee_pose_wrt_joint

        # with create_fixed_constraint(self.agent.ee_link, self.target_link):
        if True:
            move_action = RelativeLinearEEPoseAction(delta_pose_wrt_ee, self.env, 'ee', self.move_group, self.max_steps)
            move_action.check_success_input['success_fn'] = self.check_success
            succ, obs = move_action.run()
        self.datalog.extend(obs)
        if not succ:
            logging.debug('Failed to %s the slide door of the %s: failed to reach the target status' %
                             (self.target_status, self.target_obj.name))
            return False
        _, obs = GripperOpenCloseAction('open', self.env).run()
        self.datalog.extend(obs)
        return True

    def get_goal_description(self):
        return '%s the slide door of the %s' % (self.target_status, self.target_obj.name)


class RotateOperationTask(OperationTaskBase):
    def operation_step(self):
        if self.target_status == 'open':
            delta_angle = self.limit[1] - self.cur_target_qpos
        else:
            delta_angle = self.limit[0] - self.cur_target_qpos

        with create_fixed_constraint(self.agent.ee_link, self.target_link):
            move_action = CircularEEPoseAction(self.target_joint, delta_angle, self.env, self.move_group, self.max_steps)
            move_action.check_success_input['success_fn'] = self.check_success
            succ, obs = move_action.run()
        self.datalog.extend(obs)
        if not succ:
            logging.debug('Failed to %s the rotate door of the %s: failed to reach the target status' %
                             (self.target_status, self.target_obj.name))
            return False
        return True

    def get_goal_description(self):
        return '%s the rotate door of the %s' % (self.target_status, self.target_obj.name)


class PushButtonTask(BasicAction):
    # todo: push button
    pass
