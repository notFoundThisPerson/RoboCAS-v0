import hydra
from omegaconf import OmegaConf
import numpy as np
import os
from sapien.core import Pose
from transforms3d.euler import euler2quat

from robot_sim.tasks.data_collection_task import DataCollectionTask
from robot_sim.tasks.basic_tasks.move_and_grasp_task import MoveAndGraspTask
from robot_sim.tasks.basic_actions.trajectory_actions import MoveToPoseAction, RelativeLinearEEPoseAction
# from robot_sim.tasks.basic_actions.navigation_actions import NavigateAction
from robot_sim.tasks.basic_actions.gripper_actions import GripperOpenCloseAction, GraspAction
from robot_sim.tasks.basic_tasks.operation_tasks import SlideOperationTask
from robot_sim.utils import get_nearest_unique_obj, extend_goal_to_log, GoalTypes


class GraspAndConveyTask(DataCollectionTask):
    def init_subtask_list(self):
        global retry_times, target_obj, support_obj, grasped_obj_list
        retry_times, target_obj, support_obj = self.retry_times, None, None
        grasped_obj_list = []

        def init_episode_collection():
            global retry_times, target_obj, support_obj, grasped_obj_list
            basket = self.env.get_articulation_by_name('basket')
            if np.linalg.norm(basket.pose.p - basket.config.initial_pose.p) > 0.1 or \
                    basket.pose.to_transformation_matrix()[2, 2] < np.cos(np.deg2rad(20)):
                return False

            retry_times = self.retry_times
            target_obj = None
            obj_list = [obj for obj in self.env.objs if obj not in grasped_obj_list]
            unique_objs, _ = get_nearest_unique_obj(obj_list, self.agent.base_pose_world.p)
            target_obj = np.random.choice(unique_objs, 1)[0]
            grasped_obj_list.append(target_obj)
            # support_obj = self.env.get_support_articulation(target_obj)
            support_obj = self.env.get_articulation_by_name('table')
            if support_obj is None or support_obj not in self.env.articulations:
                return False
            self.lang_goal = 'grasp the %s on the %s and move it to the basket' % (target_obj.name, support_obj.name)
            return True

        def grasp_stage():
            global retry_times, target_obj, support_obj
            cur_scene = self.scene.pack()
            grasps = self.sample_grasp_pose(target_obj, max_samples=retry_times)
            for grasp_pose in grasps:
                if support_obj.name == 'shelf':
                    after_grasp_offset_base = Pose([-0.1, 0, 0.02])
                else:
                    after_grasp_offset_base = Pose([0, 0, 0.1])
                succ, obs = MoveAndGraspTask(target_obj, grasp_pose, self.env, move_group='arm',
                                             after_grasp_offset_base=after_grasp_offset_base).run()
                if succ:
                    obs = extend_goal_to_log(obs, 'grasp the %s on the %s' % (target_obj.name, support_obj.name),
                                             GoalTypes.MOVE_AND_GRASP)
                    self.data_log.extend(obs)
                    return True
                retry_times -= 1
                self.scene.unpack(cur_scene)
            return False

        def drop_stage():
            global retry_times, target_obj
            cur_scene = self.scene.pack()
            drop_pose = self.env.get_articulation_by_name('basket').pose.to_transformation_matrix()
            drop_pose[2, 3] += 0.25
            drop_pose[:3, 1] = np.array([0, 0, 1])
            drop_pose[:3, 2] = np.array([*(drop_pose[:2, 3] - self.agent.base_pose_world.p[:2]), 0])
            drop_pose[:3, 2] /= np.linalg.norm(drop_pose[:3, 2])
            drop_pose[:3, 0] = np.cross(drop_pose[:3, 1], drop_pose[:3, 2])

            rand_pose_pos_bounds = [[-0.1, 0, -0.1], [0.1, 0.1, 0]]
            rand_pose_rot_bounds = [[30, -10, -5], [70, 10, 5]]

            while retry_times > 0:
                rand_shift_pos = np.random.uniform(rand_pose_pos_bounds[0], rand_pose_pos_bounds[1], [3])
                rand_shift_rot = np.random.uniform(rand_pose_rot_bounds[0], rand_pose_rot_bounds[1], [3])
                rand_shift_rot = euler2quat(*np.deg2rad(rand_shift_rot), 'sxyz')
                rand_shift = Pose(rand_shift_pos, rand_shift_rot).to_transformation_matrix()
                rand_drop_pose = np.matmul(drop_pose, rand_shift)

                # cur_base_pos = self.agent.base_pose_world.p[:2]
                # obj_pos = rand_drop_pose[:2, 3]
                # robot_to_obj_vec = obj_pos - cur_base_pos
                # dist = np.linalg.norm(robot_to_obj_vec)
                # if dist > 0.6:
                #     vec_x, vec_y = robot_to_obj_vec / dist
                #     angles = sorted(list(range(-40, 45, 5)), key=lambda x: abs(x))
                #     for ang in angles:
                #         ang = np.deg2rad(ang)
                #         sinx, cosx = np.sin(ang), np.cos(ang)
                #         vec = np.array([vec_x * cosx - vec_y * sinx, vec_x * sinx + vec_y * cosx])
                #         stand_point = obj_pos - vec * 0.6
                #         face_angle = np.arctan2(vec[1], vec[0])
                #
                #         succ, obs = NavigateAction(stand_point, face_angle, self.env, attached_obj_list=[target_obj]).run()
                #         if succ:
                #             data_log.extend(obs)
                #             break

                succ, obs = MoveToPoseAction(rand_drop_pose, 'close', self.env, 'world', move_group='arm',
                                             attached_obj_list=[target_obj]).run()
                if succ and GraspAction(target_obj, self.env).check_grasped():
                    goal = 'Drop the %s into the basket' % target_obj.name
                    obs = extend_goal_to_log(obs, goal, GoalTypes.TRANSFER_OBJECT)
                    self.data_log.extend(obs)
                    _, obs = GripperOpenCloseAction('open', self.env).run()
                    obs = extend_goal_to_log(obs, goal, GoalTypes.TRANSFER_OBJECT)
                    self.data_log.extend(obs)
                    _, obs = RelativeLinearEEPoseAction(Pose([0, 0, 0.05]), self.env, 'world', max_steps=10).run()
                    obs = extend_goal_to_log(obs, goal, GoalTypes.TRANSFER_OBJECT)
                    self.data_log.extend(obs)
                    return True
                retry_times -= 1
                self.scene.unpack(cur_scene)
            return False

        self.subtask_list = [init_episode_collection, grasp_stage, drop_stage]


class GraspFromCabinetTask(DataCollectionTask):
    def init_subtask_list(self):
        global retry_times, target_obj, support_obj
        retry_times, target_obj, support_obj = self.retry_times, None, None

        def init_episode_collection():
            global retry_times, target_obj, support_obj
            retry_times = self.retry_times
            unique_objs, _ = get_nearest_unique_obj(self.env.objs, self.agent.base_pose_world.p)
            target_obj = np.random.choice(unique_objs, 1)[0]
            support_obj = self.env.get_support_articulation(target_obj)
            if support_obj is None or support_obj not in self.env.articulations:
                return False
            self.lang_goal = 'grasp the %s on the %s and move it to the basket' % (target_obj.name, support_obj.name)
            return True
        
        def open_drawer_stage():
            global retry_times, support_obj
            cur_scene = self.scene.pack()
            while retry_times > 0:
                succ, obs = SlideOperationTask(support_obj, support_obj.operate_tasks[0], 'open', self.env, 'arm').run()
                if succ:
                    obs = extend_goal_to_log(obs, 'open the %s' % support_obj.name, GoalTypes.OPERATE_OBJECT)
                    self.data_log.extend(obs)
                    return True
                retry_times -= 1
                self.scene.unpack(cur_scene)
            return False

        def grasp_stage():
            global retry_times, target_obj
            
            cur_scene = self.scene.pack()
            grasps = self.sample_grasp_pose(target_obj, max_samples=retry_times)
            for grasp_pose in grasps:
                if support_obj.name == 'shelf':
                    after_grasp_offset_base = Pose([-0.1, 0, 0.02])
                else:
                    after_grasp_offset_base = Pose([0, 0, 0.1])
                succ, obs = MoveAndGraspTask(target_obj, grasp_pose, self.env, move_group='arm',
                                             after_grasp_offset_base=after_grasp_offset_base).run()
                if succ:
                    obs = extend_goal_to_log(obs, 'grasp the %s on the %s' % (target_obj.name, support_obj.name),
                                             GoalTypes.MOVE_AND_GRASP)
                    self.data_log.extend(obs)
                    return True
                retry_times -= 1
                self.scene.unpack(cur_scene)
            return False

        def drop_stage():
            global retry_times, target_obj
            cur_scene = self.scene.pack()
            drop_pose = self.env.get_articulation_by_name('basket').pose.to_transformation_matrix()
            drop_pose[2, 3] += 0.3
            drop_pose[:3, 2] = np.array([0, 0, -1])
            drop_pose[:3, 1] = np.array([*(drop_pose[:2, 3] - self.agent.base_pose_world.p[:2]), 0])
            drop_pose[:3, 1] /= np.linalg.norm(drop_pose[:3, 1])
            drop_pose[:3, 0] = np.cross(drop_pose[:3, 1], drop_pose[:3, 2])

            while retry_times > 0:
                succ, obs = MoveToPoseAction(drop_pose, 'close', self.env, 'world', move_group='arm',
                                             attached_obj_list=[target_obj]).run()
                if succ and GraspAction(target_obj, self.env).check_grasped():
                    goal = 'Drop the %s into the basket' % target_obj.name
                    obs = extend_goal_to_log(obs, goal, GoalTypes.TRANSFER_OBJECT)
                    self.data_log.extend(obs)
                    _, obs = GripperOpenCloseAction('open', self.env).run()
                    obs = extend_goal_to_log(obs, goal, GoalTypes.TRANSFER_OBJECT)
                    self.data_log.extend(obs)
                    _, obs = RelativeLinearEEPoseAction(Pose([0, 0, 0.05]), self.env, 'world', max_steps=10).run()
                    obs = extend_goal_to_log(obs, goal, GoalTypes.TRANSFER_OBJECT)
                    self.data_log.extend(obs)
                    return True
                retry_times -= 1
                self.scene.unpack(cur_scene)
            return False

        self.subtask_list = [init_episode_collection, open_drawer_stage, grasp_stage, drop_stage]


@hydra.main(os.path.join(os.path.dirname(__file__), '../config'), 'grasp_and_convey_table', '1.3')
def main(conf: OmegaConf):
    task = hydra.utils.instantiate(conf.task)
    task.run(conf.num_episodes)


if __name__ == '__main__':
    main()
