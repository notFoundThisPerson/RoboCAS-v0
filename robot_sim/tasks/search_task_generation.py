import hydra
from omegaconf import OmegaConf
import numpy as np
import os
from sapien.core import Pose
from transforms3d.euler import euler2quat, euler2mat
from transforms3d.quaternions import quat2mat, mat2quat
import logging
import cv2
from scipy.spatial import KDTree

from robot_sim.tasks.data_collection_task import DataCollectionTask
from robot_sim.tasks.basic_tasks.move_and_grasp_task import MoveAndGraspTask
from robot_sim.tasks.basic_actions.trajectory_actions import MoveToPoseAction, RelativeLinearEEPoseAction
from robot_sim.tasks.basic_actions.gripper_actions import GripperOpenCloseAction, GraspAction
from robot_sim.utils import extend_goal_to_log, GoalTypes, DataLog


class SearchAndGraspTask(DataCollectionTask):
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
            # todo: fix hard code
            support_obj = self.env.get_articulation_by_name('table')

            def get_occluded_objs():
                _, ub = support_obj.get_AABB()
                cam_location = support_obj.pose.p + np.array([0, 0, ub[2] + 0.5])
                tmp_cam = self.scene.add_camera('tmp', 512, 512, fovy=np.pi / 2, far=1, near=1e-3)
                tmp_cam.set_pose(Pose(cam_location, euler2quat(-np.pi / 2, np.pi / 2, 0)))
                for link in self.agent.robot.get_links():
                    link.hide_visual()

                def get_mask():
                    self.scene.update_render()
                    tmp_cam.take_picture()
                    return tmp_cam.get_uint32_texture('Segmentation')[..., 1].astype(np.uint8)

                visual_ratio = []
                full_seg = get_mask()
                for obj in self.env.objs:
                    obj.model.hide_visual()
                for obj in obj_list:
                    obj.model.unhide_visual()
                    seg = get_mask()
                    obj.model.hide_visual()
                    full_area = np.sum(seg == obj.model.get_id())
                    if full_area == 0:
                        continue
                    visible_area = np.sum(full_seg == obj.model.get_id())
                    visual_ratio.append(visible_area / full_area)
                for obj in self.env.objs:
                    obj.model.unhide_visual()
                for link in self.agent.robot.get_links():
                    link.unhide_visual()
                # select occluded target
                occ_objs = [obj for obj, ratio in zip(obj_list, visual_ratio) if ratio < 0.9]
                self.scene.remove_camera(tmp_cam)
                return occ_objs

            obj_list = get_occluded_objs()
            if len(obj_list) == 0:
                logging.debug('init_episode_collection: can not find occluded objs')
                return False
            dist = [np.linalg.norm(obj.pose.p[:2] - self.agent.robot.pose.p[:2]) for obj in obj_list]
            weight = 1 / np.array(dist)
            target_obj = np.random.choice(obj_list, p=weight / np.sum(weight))
            grasped_obj_list.append(target_obj)
            if support_obj is None or support_obj not in self.env.articulations:
                logging.debug(f'init_episode_collection: support_obj({support_obj.name}) is None or not in env')
                return False
            self.lang_goal = 'search the %s from the objects on the %s and transfer it to the basket' \
                             % (target_obj.name, support_obj.name)
            logging.info(self.lang_goal)
            return True

        def search_stage():
            global support_obj, retry_times, target_obj
            obs_camera = self.env.cameras['gripper_camera']
            log = DataLog()

            def get_cur_img():
                imgs = obs_camera.get_images(True)
                return imgs['Color'][..., :3], imgs['Position'], imgs['Segmentation'][..., 1].astype(np.uint8)

            surf = np.array(list(support_obj.surfaces.values())[0])  # todo: fix hard code
            observe_center = (surf[0] + surf[1]) / 2
            observe_center = np.matmul(quat2mat(support_obj.pose.q), observe_center) + support_obj.pose.p
            observe_dist = np.random.uniform(0.4, 0.5)
            robot_stand_pose = self.agent.base_pose_world
            observe_dir = robot_stand_pose.p - observe_center
            observe_dir[2] = 0
            observe_dir = observe_dir / np.linalg.norm(observe_dir)

            while retry_times > 0:
                cur_state = self.scene.pack()
                while retry_times > 0:
                    observe_tf = np.stack([np.cross([0, 0, -1], observe_dir), np.array([0, 0, -1]), observe_dir], 1)
                    observe_angle = np.deg2rad(np.random.uniform([60, -20, 0], [90, 20, 0]))
                    observe_tf = np.matmul(euler2mat(*observe_angle, 'sxyz').T, observe_tf)
                    observe_point = observe_center + observe_tf[:3, 2] * observe_dist
                    observe_pose = Pose(observe_point, mat2quat(observe_tf)) * Pose(q=euler2quat(np.pi, 0, 0, 'sxyz'))
                    succ, obs = MoveToPoseAction(observe_pose, 'open', self.env, 'world').run()
                    if succ:
                        log.extend(obs)
                        break
                    retry_times -= 1
                    self.scene.unpack(cur_state)
                if retry_times == 0:
                    return False

                for obj in self.env.objs:
                    if obj != target_obj:
                        obj.model.hide_visual()
                _, _, full_seg = get_cur_img()
                for obj in self.env.objs:
                    obj.model.unhide_visual()
                _, cur_cloud, cur_seg = get_cur_img()

                cur_target_mask = cur_seg == target_obj.model.get_id()
                full_target_mask = full_seg == target_obj.model.get_id()
                cur_target_area = np.sum(cur_target_mask)
                full_target_area = np.sum(full_target_mask)
                if full_target_area == 0:
                    return False
                occluded_target_area = full_target_area - cur_target_area

                valid_depth_mask = cur_cloud[..., 3] < 1
                cam_model_matrix = obs_camera.camera.get_model_matrix()
                cur_cloud = np.matmul(cur_cloud[..., :3], cam_model_matrix[:3, :3].T) + cam_model_matrix[:3, 3]

                all_obj_ids = [obj.model.get_id() for obj in self.env.objs]
                occluded_mask = cv2.dilate(full_target_mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)

                occluded_object_ids = np.unique(cur_seg[occluded_mask])
                occluded_object_ids = [obj_id for obj_id in occluded_object_ids
                                       if obj_id in all_obj_ids and obj_id != target_obj.model.get_id()]
                logging.debug('occluded_object_ids: ' + str(occluded_object_ids))

                tmp = []
                target_cloud = target_obj.get_point_cloud()
                target_mean_z = np.mean(target_cloud[:, 2])
                target_tree = KDTree(target_cloud)
                for obj_id in occluded_object_ids:
                    if np.sum(np.logical_xor(full_target_mask, cur_seg == obj_id)) > 0:
                        tmp.append(obj_id)
                        continue
                    occ_obj = self.env.objs[all_obj_ids.index(obj_id)]
                    occ_obj_cloud = occ_obj.get_point_cloud()
                    occ_obj_cloud = occ_obj_cloud[occ_obj_cloud[:, 2] >= target_mean_z]
                    if len(occ_obj_cloud) == 0:
                        continue
                    occ_tree = KDTree(occ_obj_cloud)
                    nearby_idx = target_tree.query_ball_tree(occ_tree, 0.01)
                    nearby_idx = sum(nearby_idx, [])
                    if len(nearby_idx) > 0:
                        tmp.append(obj_id)
                occluded_object_ids = tmp

                if occluded_target_area / full_target_area <= 0.2 and len(occluded_object_ids) < 2:
                    log = extend_goal_to_log(log, 'find the %s on the %s' % (target_obj.name, support_obj.name),
                                             GoalTypes.FIND_OBJECT)
                    self.data_log.extend(log)
                    for _ in range(int(0.5 * self.env.sim_timestep)):
                        self.scene.step()
                    return True

                heights = [np.mean(cur_cloud[cur_seg == obj_id, 2]) for obj_id in occluded_object_ids]
                to_remove_id = occluded_object_ids[np.argmax(heights)]
                occluded_mask = cur_seg == to_remove_id
                occluded_obj_cloud = cur_cloud[np.logical_and(occluded_mask, valid_depth_mask)]

                cloud_center = np.mean(occluded_obj_cloud, axis=0)
                push_direction = cloud_center - target_obj.pose.p
                push_direction[2] = 0
                dir_len = np.linalg.norm(push_direction)
                if dir_len < 0.01:
                    push_direction = np.matmul(euler2mat(0, 0, np.deg2rad(np.random.uniform(-30, 30)), 'sxyz'), -observe_dir)
                else:
                    push_direction /= dir_len
                occluded_obj_cloud_proj = occluded_obj_cloud[:, :2]
                occluded_obj_cloud_proj = occluded_obj_cloud_proj - cloud_center[np.newaxis, :2]
                dist_to_push_line = np.abs(np.cross(occluded_obj_cloud_proj, push_direction[:2]))
                push_area_mask = dist_to_push_line < 0.02
                occluded_obj_cloud_proj = occluded_obj_cloud_proj[push_area_mask]
                proj_dist_to_center = np.dot(occluded_obj_cloud_proj, push_direction[:2])
                push_dist = np.min(proj_dist_to_center) - 0.02
                # to_remove_obj = self.env.objs[all_obj_ids.index(to_remove_id)]
                # lb, ub = to_remove_obj.get_AABB()
                # push_z = lb[2] + (ub[2] - lb[2]) / 4
                push_z = np.min(occluded_obj_cloud[push_area_mask, 2])
                push_point = np.array([*(cloud_center[:2] + push_dist * push_direction[:2]), push_z])

                # push_rotation = np.stack([np.cross(push_direction, [0, 0, -1]), push_direction, np.array([0, 0, -1])], 1)
                base_to_pose_vec = push_point - self.agent.base_pose_world.p
                base_to_pose_vec[2] = 0
                base_to_pose_vec = base_to_pose_vec / np.linalg.norm(base_to_pose_vec)
                push_rotation = np.stack([np.cross(base_to_pose_vec, [0, 0, -1]), base_to_pose_vec, np.array([0, 0, -1])], 1)
                push_start_pose = Pose(push_point, mat2quat(push_rotation))
                pre_push_offset = Pose([0, 0, np.max(occluded_obj_cloud[:, 2]) - push_z + 0.02])
                succ, obs = MoveToPoseAction(push_start_pose * pre_push_offset.inv(), 'close', self.env, 'world').run()
                if not succ:
                    retry_times -= 1
                    continue
                log.extend(obs)
                _, obs = RelativeLinearEEPoseAction(pre_push_offset, self.env, max_steps=10).run()
                log.extend(obs)
                _, obs = RelativeLinearEEPoseAction(Pose(push_direction * (np.abs(push_dist) + 0.1)), self.env, 'world', max_steps=10).run()
                log.extend(obs)
                _, obs = RelativeLinearEEPoseAction(Pose([0, 0, -0.05]), self.env, max_steps=10).run()
                log.extend(obs)
                observe_center = target_obj.pose.p
                observe_dist = np.random.uniform(0.2, 0.3)
            return False

        def grasp_stage():
            global retry_times, target_obj, support_obj
            cur_scene = self.scene.pack()
            grasps = self.sample_grasp_pose(target_obj, max_samples=retry_times)
            for grasp_pose in grasps:
                after_grasp_offset_base = Pose([0, 0, 0.15])
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
            rand_pose_rot_bounds = [[30, -10, -5], [60, 10, 5]]

            while retry_times > 0:
                rand_shift_pos = np.random.uniform(rand_pose_pos_bounds[0], rand_pose_pos_bounds[1], [3])
                rand_shift_rot = np.random.uniform(rand_pose_rot_bounds[0], rand_pose_rot_bounds[1], [3])
                rand_shift_rot = euler2quat(*np.deg2rad(rand_shift_rot), 'sxyz')
                rand_shift = Pose(rand_shift_pos, rand_shift_rot).to_transformation_matrix()
                rand_drop_pose = np.matmul(drop_pose, rand_shift)

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

        self.subtask_list = [
            init_episode_collection,
            search_stage,
            grasp_stage,
            drop_stage
        ]

    def sample_grasp_pose(self, obj, max_samples=100):
        cur_arm_base_pose = self.agent.base_pose_world
        grasp_labels, _, scores = obj.get_grasps_in_cur_scene()
        base_to_obj_vec_proj = grasp_labels[:, :2, 3] - cur_arm_base_pose.p[np.newaxis, :2]
        base_to_obj_vec_proj /= np.linalg.norm(base_to_obj_vec_proj, axis=1, keepdims=True)

        grasp_dir_proj = grasp_labels[:, :2, 2]
        grasp_dir_proj /= np.linalg.norm(grasp_dir_proj, axis=1, keepdims=True)
        grasp_dir_cos_ang = np.sum(base_to_obj_vec_proj * grasp_dir_proj, 1)
        grasp_rel_dir_flag = grasp_dir_cos_ang >= 0
        grasp_z_dir_flag = grasp_labels[:, 2, 2] < 0
        grasp_flag = np.logical_and(grasp_z_dir_flag, grasp_rel_dir_flag)
        grasp_labels = grasp_labels[grasp_flag]

        obj_center = np.mean(obj.get_point_cloud(), 0, keepdims=True)
        dist_to_center = np.linalg.norm(grasp_labels[:, :3, 3] - obj_center, axis=1)
        idx = np.argsort(dist_to_center)
        grasp_labels = grasp_labels[idx]

        cur_ee_pose = self.agent.ee_pose_world.to_transformation_matrix()
        y_axis_cos_angle = np.sum(grasp_labels[:, :3, 1] * cur_ee_pose[np.newaxis, :3, 1], 1)
        grasp_labels[y_axis_cos_angle < 0, :3, :2] *= -1

        grasp_labels = self.filter_grasp_collision(obj, grasp_labels, max_samples)
        return grasp_labels     # iterator


@hydra.main(os.path.join(os.path.dirname(__file__), '../config'), 'search_and_grasp_table', '1.3')
def main(conf: OmegaConf):
    task = hydra.utils.instantiate(conf.task)
    task.run(conf.num_episodes)


if __name__ == '__main__':
    main()
