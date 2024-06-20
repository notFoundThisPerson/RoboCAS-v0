import json
import os
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
from functools import lru_cache
from typing import Optional, List
from tqdm import tqdm


def build_dataset(data_path, time_sequence_length=6, num_train_episode=-1, num_val_episode=-1,
                  cam_view=["static_rgb"], **kwargs):
    pre_load_imgs_train = kwargs.pop('pre_load_imgs_train', False)
    pre_load_imgs_test = kwargs.pop('pre_load_imgs_test', False)
    kwargs.setdefault('pre_processed', False)
    train_dataset = MtSimDataset(data_path, 'train', time_sequence_length, num_train_episode, cam_view,
                                 pre_load_imgs=pre_load_imgs_train, **kwargs)
    val_dataset = MtSimDataset(data_path, 'val', time_sequence_length, num_val_episode, cam_view,
                               pre_load_imgs=pre_load_imgs_test, **kwargs)
    return train_dataset, val_dataset


def depth_to_cloud(depth, intrinsic):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (u - intrinsic[2]) / intrinsic[0] * z
    y = (v - intrinsic[3]) / intrinsic[1] * z
    cloud = np.stack([x, y, z], axis=2)
    return cloud


def normalize_point_cloud(cloud):
    mins = np.min(cloud, axis=(0, 1), keepdims=True)
    maxs = np.max(cloud, axis=(0, 1), keepdims=True)
    range = maxs - mins
    max_range = np.max(range)
    cloud_norm = (cloud - mins) / (max_range + 1e-8)
    return cloud_norm


class DatasetHelper(Dataset):
    def __init__(self, img_path, img_info, transform, pre_processed=False):
        super().__init__()
        assert len(img_path) == len(img_info)
        self.img_path = img_path
        self.img_info = img_info
        self.transform = transform
        self.pre_processed = pre_processed

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        if self.pre_processed:
            img = Image.open(self.img_path[idx])
        else:
            img = self.transform(Image.open(self.img_path[idx]))
        return self.img_info[idx], np.array(img, dtype=np.uint8)


class MtSimDataset(Dataset):
    def __init__(self, data_path, split='train', time_sequence_length=6, num_episode=-1, cam_view=['static_rgb'],
                 output_img_size=(320, 256), use_global_goal=True, step_types: Optional[List] = None, pre_load_imgs=True,
                 pre_processed=False):
        '''
        :param data_path: Path to the dataset
        :param split: Split of the dataset
        :param time_sequence_length: Length of the input sequence to the model
        :param num_episode: Maximum number of trajectories
        :param cam_view: List of used camera images
        :param output_img_size: Size of the output image
        :param use_global_goal: Whether to use the global goal instruction
        :param step_types: Types of sub-tasks
        '''
        super(MtSimDataset, self).__init__()
        self._data_path = data_path
        with open(os.path.join(self._data_path, 'data_info.json'), 'r') as f:
            info = json.load(f)
        assert split in ['train', 'val', 'test']
        self._episode_idx = info[split + '_eps']
        if num_episode > 0:
            self._episode_idx = self._episode_idx[:num_episode]

        self.data_list = []
        if use_global_goal:
            for idx in self._episode_idx:
                ep_length = info['ep_length'][str(idx)]
                end_frame_idx = list(range(ep_length))
                # if np.random.rand() < 0.5:
                #     # Repeat the grasping and picking process
                #     gripper_traj = np.load(os.path.join(self._data_path, 'episode_%07d/episode_info.npz' % idx))
                #     gripper_traj = gripper_traj['gripper_width'] > 0.04
                #     grasp_data_idx = np.min(np.where(np.logical_not(gripper_traj))[0])
                #     end_frame_idx += list(range(max(grasp_data_idx - 5, 0), min(grasp_data_idx + 5, ep_length)))
                for end_frame in end_frame_idx:
                    start_frame = max(end_frame - time_sequence_length + 1, 0)
                    self.data_list.append((idx, start_frame, end_frame, 0, ep_length - 1))
        else:
            for idx in self._episode_idx:
                for step_goal in info['step_goals'][str(idx)]:
                    goal = step_goal['goal_description']
                    start, end = step_goal['goal_range']

                    if step_types is not None and goal not in step_types:
                        continue
                    end_frame_idx = list(range(start, end + 1))
                    if goal == 'move and grasp' and np.random.rand() < 0.5:
                        gripper_traj = np.load(os.path.join(self._data_path, 'episode_%07d/episode_info.npz' % idx))
                        gripper_traj = gripper_traj['gripper_status']
                        grasp_data_idx = np.min(np.where(np.logical_not(gripper_traj))[0])
                        end_frame_idx += list(range(max(grasp_data_idx - 5, start), min(grasp_data_idx + 5, end + 1)))
                    for end_frame in end_frame_idx:
                        start_frame = max(end_frame - time_sequence_length + 1, start)
                        self.data_list.append((idx, start_frame, end_frame, start, end))

        self._time_sequence_length = time_sequence_length
        for view in cam_view:
            cam, type = view.split('_')
            assert cam in ['static', 'base', 'gripper'] and type in ['rgb', 'depth']
        self._views = cam_view
        self._img_width, self._img_height = output_img_size
        self._use_global_goal = use_global_goal
        self.transform = Resize((self._img_height, self._img_width))

        self.img_buffer = {}
        self.pre_processed = pre_processed
        print('use pre_processed data: ', pre_processed)
        if pre_load_imgs:
            print('loading images')
            img_paths, img_info = [], []

            def fn(ep_idx, img_idx):
                for view in cam_view:
                    cam, type = view.split('_')
                    if self.pre_processed:
                        img_path = os.path.join(self._data_path,
                                                'episode_%07d/%s_camera/%s_processed/%04d.png' % (ep_idx, cam, type, img_idx))
                    else:
                        img_path = os.path.join(self._data_path,
                                                'episode_%07d/%s_camera/%s/%04d.png' % (ep_idx, cam, type, img_idx))
                    img_paths.append(img_path)
                    img_info.append((ep_idx, cam, type, img_idx))

            if use_global_goal or step_types is None:
                for idx in self._episode_idx:
                    for i in range(info['ep_length'][str(idx)]):
                        fn(idx, i)
            else:
                for idx in self._episode_idx:
                    for goal, (start, end) in info['step_goal_type'][str(idx)].items():
                        if goal not in step_types:
                            continue
                        for i in range(start, end + 1):
                            fn(idx, i)
            helper_dataset = DatasetHelper(img_paths, img_info, self.transform)
            helper_loader = DataLoader(helper_dataset, batch_size=10, num_workers=10, shuffle=False, drop_last=False,
                                       collate_fn=lambda data: ([d[0] for d in data], [d[1] for d in data]))
            for imgs_info, imgs in tqdm(helper_loader):
                for info, img in zip(imgs_info, imgs):
                    self.img_buffer[info] = img

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # t0 = time()
        imgs = self.get_image(idx)
        episode_data = self.get_episode_data(idx)
        terminate_episode = self.get_episode_status(idx)
        sample_obs = {
            "image": imgs.float().permute(0, 3, 1, 2),
            "natural_language_embedding": torch.from_numpy(episode_data['natural_language_emb']).float(),
            "joint_position": torch.from_numpy(episode_data['joint']).float(),
            "position": torch.from_numpy(episode_data['ee_pos']).float(),
            "orientation": torch.from_numpy(episode_data['ee_orn']).float()
        }
        sample_action = {
            "world_vector": torch.from_numpy(episode_data['rel_pos']).float(),
            "rotation_delta": torch.from_numpy(episode_data['rel_orn']).float(),
            "gripper_closedness_action": torch.from_numpy(episode_data['gripper_cmd']).float(),
            "terminate_episode": torch.from_numpy(terminate_episode.argmax(-1)).long(),
        }
        # print('load one traj: ', time() - t0)
        return sample_obs, sample_action

    def get_episode_status(self, idx):
        episode_idx, start_frame_idx, end_frame_idx, traj_start_idx, traj_end_idx = self.data_list[idx]
        pad_step_num = self._time_sequence_length - (end_frame_idx - start_frame_idx + 1)
        episode_status = [np.zeros([pad_step_num, 4], dtype=np.int32)]
        for i in range(start_frame_idx, end_frame_idx + 1):
            status = np.array([i == 0, i not in [traj_start_idx, traj_end_idx], i == traj_end_idx, 0], dtype=np.int32)
            episode_status.append(status[np.newaxis])
        episode_status = np.concatenate(episode_status, axis=0)
        return episode_status

    def get_episode_data(self, idx):
        episode_idx, start_frame_idx, end_frame_idx, _, traj_end_idx = self.data_list[idx]
        episode_path = os.path.join(self._data_path, 'episode_%07d' % episode_idx)
        episode_info = self._load_episode_data(episode_path)

        episode_data = dict(rel_pos=episode_info['rel_pos'],
                            rel_orn=episode_info['rel_orn'],
                            joint=episode_info['robot_joints'],
                            gripper_status=episode_info['gripper_status'].astype(float).reshape(-1, 1),
                            ee_pos=episode_info['ee_pos'],
                            ee_orn=episode_info['ee_orn'])
        if not self._use_global_goal:
            episode_data['natural_language_emb'] = episode_info['step_goal_embs']

        # start_idx = max(start_frame_idx, 0)
        for k in episode_data.keys():
            episode_data[k] = episode_data[k][start_frame_idx:end_frame_idx + 1]

        # use the gripper_status of the next timestep as the GT action in the current timestep
        start_idx = start_frame_idx + 1
        if end_frame_idx == traj_end_idx:
            end_idx = end_frame_idx + 1
            episode_data['gripper_cmd'] = episode_info['gripper_status'].astype(float).reshape(-1, 1)[start_idx:end_idx]
            # longer than the length of current trajectory
            episode_data['gripper_cmd'] = np.concatenate([episode_data['gripper_cmd'], episode_data['gripper_cmd'][-1:]], 0)
        else:
            end_idx = end_frame_idx + 2
            episode_data['gripper_cmd'] = episode_info['gripper_status'].astype(float).reshape(-1, 1)[start_idx:end_idx]
        episode_data['rel_orn'] = R.from_quat(episode_data['rel_orn']).as_euler('xyz', degrees=False)

        # number of steps from the start of the trajectory is less than time_sequence_length
        pre_pad_num = self._time_sequence_length - (end_frame_idx - start_frame_idx + 1)
        if pre_pad_num > 0:
            repeat_pad_items = ['joint', 'ee_pos', 'ee_orn', 'gripper_cmd']
            zero_pad_items = ['rel_pos', 'rel_orn']
            if 'natural_language_emb' in episode_data:
                repeat_pad_items.append('natural_language_emb')
            for k in repeat_pad_items:
                v = episode_data[k]
                episode_data[k] = np.concatenate([v[0:1].repeat(pre_pad_num, 0), v], 0)
            for k in zero_pad_items:
                v = episode_data[k]
                zero_pad = np.zeros([pre_pad_num, v.shape[1]], dtype=v.dtype)
                episode_data[k] = np.concatenate([zero_pad, v], axis=0)
        if self._use_global_goal:
            episode_data['natural_language_emb'] = \
                episode_info['language_embedding'][np.newaxis].repeat(self._time_sequence_length, axis=0)
        return episode_data

    def get_image(self, idx):
        # t0 = time()
        episode_idx, start_frame_idx, end_frame_idx, _, _ = self.data_list[idx]
        episode_path = os.path.join(self._data_path, 'episode_%07d' % episode_idx)
        imgs = [[] for _ in self._views]
        for img_idx in range(start_frame_idx, end_frame_idx + 1):
            for i, view in enumerate(self._views):
                camera, type = view.split('_')
                if type == 'rgb':
                    img = self._load_rgb_image(episode_idx, camera, img_idx, self.pre_processed)
                elif type == 'depth':
                    img = self._load_depth_image(episode_path, camera, img_idx, self.pre_processed)
                else:
                    raise ValueError
                imgs[i].append(img)
        imgs = np.concatenate([np.stack(view_imgs, 0) for view_imgs in imgs], 1)  # [N, H, W, 3]
        if imgs.shape[0] < self._time_sequence_length:
            num_repeats = self._time_sequence_length - imgs.shape[0]
            repeat_imgs = imgs[0:1].repeat(num_repeats, 0)
            imgs = np.concatenate([repeat_imgs, imgs], 0)
        # print('load image: ', time() - t0)
        return torch.from_numpy(imgs)

    @staticmethod
    @lru_cache(maxsize=128, typed=False)
    def _load_episode_data(episode_path):
        return np.load(os.path.join(episode_path, 'episode_info.npz'))

    # @lru_cache(maxsize=25000, typed=False)
    def _load_rgb_image(self, episode_idx, camera, img_idx, pre_processed=False):
        if len(self.img_buffer) > 0:
            img = self.img_buffer[(episode_idx, camera, 'rgb', img_idx)]
        else:
            if pre_processed:
                img = Image.open(os.path.join(self._data_path, 'episode_%07d/%s_camera/rgb_processed/%04d.png' % \
                                              (episode_idx, camera, img_idx)))
            else:
                img = Image.open(os.path.join(self._data_path, 'episode_%07d/%s_camera/rgb/%04d.png' % \
                                              (episode_idx, camera, img_idx)))
                img = self.transform(img)
            img = np.array(img)
        img = img / 255.0
        return img

    # @lru_cache(maxsize=25000, typed=False)
    def _load_depth_image(self, episode_idx, camera, img_idx, pre_processed=False):
        if len(self.img_buffer) > 0:
            img = self.img_buffer[(episode_idx, camera, 'depth', img_idx)]
        else:
            if pre_processed:
                img = Image.open(os.path.join(self._data_path, 'episode_%07d/%s_camera/depth_processed/%04d.png' % \
                                              (episode_idx, camera, img_idx)))
            else:
                img = Image.open(os.path.join(self._data_path, 'episode_%07d/%s_camera/depth/%04d.png' % \
                                              (episode_idx, camera, img_idx)))
                img = self.transform(img)
            img = np.array(img)
        img = img * 1e-3
        intrinsic = np.load(os.path.join(self._data_path, 'episode_%07d/%s_camera/intrinsic.npy' % (episode_idx, camera)))
        img = normalize_point_cloud(depth_to_cloud(img, intrinsic))
        return img


if __name__ == '__main__':
    _, val_dataset = build_dataset('/home/caohaiheng/Projects/datasets/shelf_grasp_data')
    loader = DataLoader(val_dataset, batch_size=2, shuffle=True)
    print(len(loader))
    for sample_obs, sample_action in loader:
        print('Observations:')
        for k, v in sample_obs.items():
            print('\t', k, v.shape, v.dtype)
        print('Actions:')
        for k, v in sample_action.items():
            print('\t', k, v.shape, v.dtype)
        break
