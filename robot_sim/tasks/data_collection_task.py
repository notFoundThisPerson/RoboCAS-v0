import numpy as np
from pathlib import Path
from typing import Union, List, Optional
import sapien.core as sapien
import xml.dom.minidom as xmldom
import cv2
from scipy.spatial.transform import Rotation as R
from sentence_transformers import SentenceTransformer
import os
from transforms3d.euler import euler2quat
import logging

from mani_skill2.agents.robots.mobile_panda import get_entities_by_names
from robot_sim.envs.basic_env import TaskBaseEnv
from robot_sim.agents.robots.mobile_franka_panda import MobileFrankaPanda
from robot_sim.utils import DataLog
from robot_sim.tasks.basic_actions import BasicAction


class DataCollectionTask:
    def __init__(self, env: TaskBaseEnv, output_data_path: Union[str, Path], random_init_prob=0.,
                 max_traj_length=100, sentence_transformer: Optional[SentenceTransformer] = None,
                 subtask_list: Optional[List[BasicAction]] = None, retry_times=20):
        self.env = env
        self.agent: MobileFrankaPanda = self.env.agent
        self.scene: sapien.Scene = self.env.scene
        # self._build_avoid_collision_matrix()
        if isinstance(output_data_path, str):
            output_data_path = Path(output_data_path)
        self.output_data_path = output_data_path
        self.output_data_path.mkdir(parents=True, exist_ok=True)
        exist_epi_trajs = os.listdir(self.output_data_path.as_posix())
        exist_epi_trajs = [int(p.split('_')[1]) for p in exist_epi_trajs if p.startswith('episode_')]
        self.traj_idx = max(exist_epi_trajs) + 1 if len(exist_epi_trajs) > 0 else 0
        self.random_init_prob = random_init_prob
        self.max_traj_length = max_traj_length
        self.sentence_transformer = sentence_transformer
        self.subtask_list = subtask_list
        self.retry_times = retry_times

        self.data_log = DataLog()
        self.lang_goal = ''
        self.init_subtask_list()

    def reset(self):
        self.env.reset()
        self.env.render()
        self.scene.step()

    def init_subtask_list(self):
        raise NotImplementedError

    def run(self, num_episodes=None):
        episode_cnt = 0
        run_cnt = 0
        init_fail_cnt = 0
        need_reset = False
        while num_episodes is None or episode_cnt <= num_episodes:
            if init_fail_cnt >= 10:
                logging.error('Task init failed')
                exit(1)
            if need_reset:  # Reset the environment every several episodes
                need_reset = False
                self.reset()
                logging.info('task reset')
            elif np.random.sample() < self.random_init_prob:
                self.sample_initial_state()

            self.data_log.clear()
            self.lang_goal = ''

            cur_scene = self.scene.pack()
            for subtask in self.subtask_list:
                logging.info('Running subtask: %s' % subtask.__name__)
                succ = subtask()
                if not succ:
                    logging.info('Subtask failed: %s' % subtask.__name__)
                    if 'init' in subtask.__name__:
                        need_reset = True
                        init_fail_cnt += 1
                    else:
                        init_fail_cnt = 0
                        self.scene.unpack(cur_scene)
                    break
            else:
                self._gather_observation(self.data_log, self.lang_goal)
                episode_cnt += 1
            run_cnt += 1
            if run_cnt >= 5:
                need_reset = True
                run_cnt = 0

    def sample_grasp_pose(self, obj, max_samples=100):
        cur_arm_base_pose = self.agent.base_pose_world
        grasp_labels, _, scores = obj.get_grasps_in_cur_scene()
        # Relative xy position between object and arm base
        base_to_obj_vec_proj = grasp_labels[:, :2, 3] - cur_arm_base_pose.p[np.newaxis, :2]
        base_to_obj_vec_proj /= np.linalg.norm(base_to_obj_vec_proj, axis=1, keepdims=True)

        grasp_dir_proj = grasp_labels[:, :2, 2]
        grasp_dir_proj /= np.linalg.norm(grasp_dir_proj, axis=1, keepdims=True)
        grasp_dir_cos_ang = np.sum(base_to_obj_vec_proj * grasp_dir_proj, 1)
        grasp_rel_dir_flag = grasp_dir_cos_ang > np.cos(np.pi / 4)
        grasp_z_dir_flag = grasp_labels[:, 2, 2] < 0
        grasp_x_dir_flag = np.abs(grasp_labels[:, 2, 0]) < np.cos(np.deg2rad(30))

        grasp_flag = np.all(np.stack([grasp_z_dir_flag, grasp_rel_dir_flag, grasp_x_dir_flag], 0), 0)
        grasp_labels = grasp_labels[grasp_flag]
        np.random.shuffle(grasp_labels)
        grasp_labels[grasp_labels[:, 2, 1] < 0, :3, :2] *= -1
        grasp_labels = self.filter_grasp_collision(obj, grasp_labels)

        output_cnt = 0
        sample_cnt = 0
        for grasp_sample in grasp_labels:
            logging.debug('Sampling grasp pose: %d' % sample_cnt)
            sample_cnt += 1
            if self._is_grasp_available(grasp_sample):
                output_cnt += 1
                logging.debug('Return the %d-th grasp sampled after %d samples' % (output_cnt, sample_cnt))
                yield grasp_sample
                sample_cnt = 0
                if output_cnt >= max_samples:
                    return

    def filter_grasp_collision(self, target_obj, grasp_poses, max_samples=0):
        cloud = self.env.get_point_cloud()

        def point_in_box_mask(pc, lower_bound, upper_bound):
            return np.all(np.concatenate([pc >= lower_bound, pc <= upper_bound], 1), 1)

        lb, ub = target_obj.get_AABB()
        nearby_cloud_flag = point_in_box_mask(cloud, lb - 0.1, ub + 0.1)
        cloud = cloud[nearby_cloud_flag]

        gripper_boxs = np.array([[[0.04, -0.01, -0.09], [0.045, 0.01, 0]],
                                 [[-0.045, -0.01, -0.09], [-0.04, 0.01, 0]],
                                 [[-0.1, -0.02, -0.18], [0.1, 0.02, -0.09]]])

        available_grasp_idx = []
        for i, pose in enumerate(grasp_poses):
            cloud_in_gripper = np.matmul(cloud - pose[np.newaxis, :3, 3], pose[:3, :3])
            num_points_collision = 0
            for box in gripper_boxs:
                num_points_collision += np.sum(point_in_box_mask(cloud_in_gripper, box[0], box[1]))
            if num_points_collision == 0:
                available_grasp_idx.append(i)
                yield pose
                if 0 < max_samples <= len(available_grasp_idx):
                    # break
                    return
        # return grasp_poses[available_grasp_idx]

    def _is_grasp_available(self, grasp):
        cur_state = self.scene.pack()
        grasp_pose = sapien.Pose().from_transformation_matrix(grasp)

        hand_urdf = (Path(__file__).parents[2] / 'assets/urdf/hand.urdf').as_posix()
        builder = self.scene.create_urdf_loader()
        builder.fix_root_link = True
        hand = builder.load(hand_urdf)
        hand.set_pose(sapien.Pose())
        hand.set_qpos(np.array([0.04, 0.04]))
        for joint in hand.get_active_joints():
            joint.set_drive_property(0, 100, 100)
            joint.set_drive_target(0.04)

        hand_links = hand.get_links()
        hand_tcp_link_rel_pose = get_entities_by_names(hand_links, 'panda_hand_tcp').pose
        hand.set_pose(grasp_pose * hand_tcp_link_rel_pose.inv())
        self.scene.step()
        contacts = self.scene.get_contacts()

        def get_minimum_distance(contact):
            d = np.inf
            for point in contact.points:
                d = min(d, point.separation)
            return d

        allow = True
        for cont in contacts:
            cont_count = int(cont.actor0 in hand_links) + int(cont.actor1 in hand_links)
            if cont_count == 1 and get_minimum_distance(cont) < 2e-3:
                allow = False
                break

        self.scene.remove_articulation(hand)
        self.scene.unpack(cur_state)
        return allow

    def allow_collision(self, link1: sapien.Actor, link2: sapien.Actor):
        return self._acm[link1.id, link2.id]

    def _build_avoid_collision_matrix(self):
        objs = self.env.scene.get_all_actors()
        objs.extend(self.agent.robot.get_links())
        num_objs = len(objs)
        acm = np.eye(num_objs + 1, dtype=bool)

        robot_srdf_path = self.agent.urdf_path.replace('.urdf', '.srdf')
        robot_acm_info = xmldom.parse(robot_srdf_path).getElementsByTagName('disable_collisions')
        for info in robot_acm_info:
            link1 = get_entities_by_names(objs, info.getAttribute('link1'))
            link2 = get_entities_by_names(objs, info.getAttribute('link2'))
            acm[link1.id, link2.id] = acm[link2.id, link1.id] = True

        ground = get_entities_by_names(objs, 'ground')
        acm[ground.id, self.agent.base_link.id] = acm[self.agent.base_link.id, ground.id] = True
        self._acm = acm

    def sample_initial_state(self):
        cur_trajs = os.listdir(self.output_data_path.as_posix())
        cur_trajs = [traj for traj in cur_trajs if traj.startswith('episode_')]
        if len(cur_trajs) == 0:
            return
        sampled_traj = np.random.choice(cur_trajs, 1)[0]
        traj_info = np.load((self.output_data_path / sampled_traj / 'episode_info.npz').as_posix())
        idx = np.random.randint(traj_info['episode_length'])

        init_qpos = [*traj_info['agv_pose'][idx], *traj_info['robot_joints'][idx]]
        self.agent.robot.set_qpos(init_qpos)

        ee_pose_base = sapien.Pose(traj_info['ee_pos'][idx], traj_info['ee_orn'][idx][[3, 0, 1, 2]])
        base_pose_world = sapien.Pose(traj_info['base_pos'][idx], traj_info['base_orn'][idx][[3, 0, 1, 2]])
        ee_pose_world = base_pose_world * ee_pose_base
        for _ in range(20):
            delta_pos = np.random.rand(3)
            delta_pos *= np.random.uniform(0.05, 0.15) / np.linalg.norm(delta_pos)
            delta_orn = np.deg2rad(np.random.uniform(-10, 10, [3]))
            delta_pose = sapien.Pose(delta_pos, euler2quat(delta_orn))
            new_pose = ee_pose_world * delta_pose
            if self._is_grasp_available(new_pose):
                succ, ik = self.agent.compute_ik(new_pose, group='arm')
                if succ == 'Success':
                    self.agent.robot.set_qpos(ik[0])
                    self.agent.robot.set_drive_target(ik[0])
                    return

    def _gather_observation(self, obs, language_goal):
        episode_dir = self.output_data_path / ('episode_%07d' % self.traj_idx)

        obs['agent']['ee_pose_world'] = obs['agent']['ee_pose_world'].reshape(-1, 4, 4)
        obs['agent']['arm_base_pose_world'] = obs['agent']['arm_base_pose_world'].reshape(-1, 4, 4)

        cameras = [cam.uid for cam in self.env.agent.config.cameras]
        num_steps = obs['image'][cameras[0]]['Color'].shape[0]
        logging.info('Collecting trajectory [length: %d, path %s, goal: %s]' % (num_steps, episode_dir, language_goal))
        for cam in cameras:
            rgb_path = episode_dir / cam / 'rgb'
            depth_path = episode_dir / cam / 'depth'
            rgb_path.mkdir(parents=True)
            depth_path.mkdir(parents=True)
            for i in range(num_steps):
                file_name = '%04d.png' % i
                rgb_img_path = rgb_path / file_name
                img = cv2.cvtColor(obs['image'][cam]['Color'][i], cv2.COLOR_RGB2BGR)
                cv2.imwrite(rgb_img_path.as_posix(), img)

                depth_img_path = depth_path / file_name
                cv2.imwrite(depth_img_path.as_posix(), obs['image'][cam]['Position'][i])

            intrinsics = obs['camera_param'][cam]['intrinsic_cv'][0]
            intrinsics = np.array([intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]])
            np.save((episode_dir / cam / 'intrinsic.npy').as_posix(), intrinsics)
            obs['camera_param'][cam]['extrinsic_cv'][0] = np.linalg.inv(obs['camera_param'][cam]['extrinsic_cv'][0])

        static_cam_pose_wrt_world = obs['camera_param']['static_camera']['extrinsic_cv'][0]
        gripper_cam_pose_wrt_ee = np.matmul(np.linalg.inv(obs['agent']['ee_pose_world'][0]),
                                            obs['camera_param']['gripper_camera']['extrinsic_cv'][0])
        base_cam_pose_wrt_arm_base = np.matmul(np.linalg.inv(obs['agent']['arm_base_pose_world'][0]),
                                               obs['camera_param']['base_camera']['extrinsic_cv'][0])
        np.save((episode_dir / 'static_camera/cam_pose_wrt_world.npy').as_posix(), static_cam_pose_wrt_world)
        np.save((episode_dir / 'gripper_camera/cam_pose_wrt_ee.npy').as_posix(), gripper_cam_pose_wrt_ee)
        np.save((episode_dir / 'base_camera/cam_pose_wrt_arm_base.npy').as_posix(), base_cam_pose_wrt_arm_base)

        def encode_language_goals():
            goals_uniq = np.unique(obs['task']['lang_goal']).tolist()
            embs_uniq = self.sentence_transformer.encode(goals_uniq)
            goal_idx = [goals_uniq.index(goal) for goal in obs['task']['lang_goal']]
            return embs_uniq[goal_idx]

        episode_info = dict(
            robot_joints=obs['agent']['arm_qpos'],
            arm_joint_vel=obs['agent']['arm_qvel'],
            base_pos=obs['agent']['arm_base_pose_world'][:, :3, 3],
            base_orn=R.from_matrix(obs['agent']['arm_base_pose_world'][:, :3, :3]).as_quat(),
            gripper_width=obs['agent']['gripper_width'],
            episode_length=num_steps,
            language_goal=language_goal,
            gripper_status=(obs['agent']['gripper_target'] > 0.04).astype(bool)
        )

        if 'task' in obs.data:
            episode_info['step_lang_goals'] = obs['task']['lang_goal']
            episode_info['step_goal_type'] = obs['task']['goal_type']

        if self.sentence_transformer is not None:
            episode_info['language_embedding'] = self.sentence_transformer.encode([language_goal])[0]
            if 'task' in obs.data:
                episode_info['step_goal_embs'] = encode_language_goals()

        # ee pose wrt arm base
        ee_pose_world = obs['agent']['ee_pose_world']
        base_pose_world = obs['agent']['arm_base_pose_world']
        ee_pose_base = np.matmul(np.linalg.inv(base_pose_world), ee_pose_world)
        episode_info['ee_pos'] = ee_pose_base[:, :3, 3]
        episode_info['ee_orn'] = R.from_matrix(ee_pose_base[:, :3, :3]).as_quat()

        # ee pose at the next time wrt current timestep
        ee_rel_pose = np.matmul(np.linalg.inv(ee_pose_base[:-1]), ee_pose_base[1:])
        ee_rel_pose = np.concatenate([ee_rel_pose, np.eye(4)[np.newaxis]], 0)
        episode_info['rel_pos'] = ee_rel_pose[:, :3, 3]
        episode_info['rel_orn'] = R.from_matrix(ee_rel_pose[:, :3, :3]).as_quat()

        base_rel_pose = np.matmul(np.linalg.inv(base_pose_world[:-1]), base_pose_world[1:])
        base_rel_pose = np.concatenate([base_rel_pose, np.eye(4)[np.newaxis]], 0)
        episode_info['base_rel_pos'] = base_rel_pose[:, :3, 3]
        episode_info['base_rel_orn'] = R.from_matrix(base_rel_pose[:, :3, :3]).as_quat()

        def xyr_to_mat(xy, r):
            sinr = np.sin(r)
            cosr = np.cos(r)
            return np.array([[cosr, -sinr, 0, xy[0]], [sinr, cosr, 0, xy[1]], [0, 0, 1, 0], [0, 0, 0, 1]])

        def mat_to_xyr(mat):
            r = np.arctan2(mat[1, 0], mat[0, 0])
            return np.array([mat[0, 3], mat[1, 3], r])

        root_pose_world = self.agent.robot.pose.to_transformation_matrix()
        agv_pose_wrt_world = []
        for i in range(num_steps):
            mat = xyr_to_mat(obs['agent']['base_pos'][i], obs['agent']['base_orientation'][i])
            mat = np.matmul(root_pose_world, mat)
            agv_pose_wrt_world.append(mat_to_xyr(mat))
        episode_info['agv_pose'] = np.stack(agv_pose_wrt_world, 0)
        episode_info['rel_agv_pose'] = np.concatenate(
            [episode_info['agv_pose'][1:] - episode_info['agv_pose'][:-1], np.zeros([1, 3])], axis=0)
        np.savez((episode_dir / 'episode_info.npz').as_posix(), **episode_info)

        self.traj_idx += 1
