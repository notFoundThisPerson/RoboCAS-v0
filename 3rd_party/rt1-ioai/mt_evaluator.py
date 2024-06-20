import os
import sys
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import cv2
from tqdm import trange
import hydra
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer

CUR_DIR_PATH = os.path.dirname(__file__)
sys.path.append(CUR_DIR_PATH)
sys.path.append(os.path.abspath(os.path.join(CUR_DIR_PATH, '../..')))
sys.path = list(set(sys.path))

from IO_evaluator import Evaluator, load_config_from_json, test_loss
from robot_sim.utils import create_fixed_constraint


def get_grasped_obj(contacts):
    left_contacts = []
    right_contacts = []
    for cnt in contacts:
        other, finger = None, None
        if 'tip' in cnt.actor0.name:
            finger = cnt.actor0
            other = cnt.actor1
        elif 'tip' in cnt.actor1.name:
            finger = cnt.actor1
            other = cnt.actor0
        if other is not None and 'finger' not in other.name and 'hand' not in other.name and 'link' not in other.name:
            for p in cnt.points:
                if p.separation < 1e-3:
                    if 'left' in finger.name:
                        left_contacts.append(other)
                    else:
                        right_contacts.append(other)
                    break
    for obj in left_contacts:
        if obj in right_contacts:
            return obj
    return None


class MtEvaluator(Evaluator):
    def __init__(self, task, ckpt_path):
        args = load_config_from_json(os.path.join(ckpt_path, os.path.basename(ckpt_path) + '.json'))
        self.args = args
        self.task = task
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.max_step = 150
        self.time_sequence_length = self.args['time_sequence_length']
        self.cam_views = self.args['cam_view']
        self.lang_emb = None
        self.goal = ''

    def calc_fk(self, obs):
        # pose = self.task.env.agent.ee_link.pose
        obs["position"] = torch.stack(self.robot_pos, dim=1).to(self.device)
        obs["orientation"] = torch.stack(self.robot_orn, dim=1).to(self.device)
        return obs

    def get_language_emb(self, obs, network):
        obs["natural_language_embedding"] = self.lang_emb.reshape(1, 1, -1).repeat([1, network._time_sequence_length, 1])
        return obs

    def val(self, resume_from_checkpoint, ckpt_dir, proc_name, cam_views, test_num, results_fn, epoch, gpu_name,
            using_pos_and_orn, tar_obj_pos_rot):
        self._init_val(resume_from_checkpoint, ckpt_dir)
        self.task.set_record_path(os.path.join(os.path.dirname(results_fn), 'vid_log'))
        for idx in range(test_num):
            lift_count = 0
            self.reset()

            obj_heights = {}
            for obj in self.task.env.objs:
                obj_heights[obj.model.id] = obj.model.pose.p[2]

            constraint = None
            for _ in trange(self.max_step):
                self.update_obs()
                imgs = self.imgs
                joints = self.joints
                (terminate_episode, delta_ee_pos, delta_ee_rot, gripper_cmd,) = self.inference(self.network, imgs, joints)
                self.task.step(delta_ee_pos, delta_ee_rot, gripper_cmd)

                if gripper_cmd[0] < 0.5 and constraint is None:
                    grasped = get_grasped_obj(self.task.env.scene.get_contacts())
                    if grasped is not None and grasped.pose.p[2] >= obj_heights[grasped.id] + 0.03:
                        constraint = create_fixed_constraint(grasped, self.task.env.agent.ee_link)
                elif gripper_cmd[0] > 0.5 and constraint is not None:
                    constraint.release()
                    constraint = None
                    print(self.task.env.agent.robot.get_qpos()[3:-2])

                episode_succ = self.task.check_task_succeed(2)
                if episode_succ[0] and episode_succ[1]:
                    lift_count += 1
                    if lift_count >= 5:
                        print("dropped")
                        break
            self.write_results(results_fn, episode_succ)
            used_cams = list(set([view.split('_')[0] + '_camera' for view in self.cam_views]))
            self.task.record_episode_obs(used_cams, self.goal)

    def reset(self):
        self.task.reset()
        self.reset_obs()
        self.goal = 'grasp the %s on the table and move it to the basket' % self.task.target_obj.name
        self.lang_emb = torch.from_numpy(self.encoder.encode([self.goal])).to(self.device)
        print(self.goal)

    def reset_obs(self):
        self.imgs = []
        self.joints = [torch.zeros(1, 7)] * self.time_sequence_length  # Not used
        self.robot_pos = []  # Padding 3D position into 4D vector
        self.robot_orn = []

    def update_obs(self):
        cur_obs = self.task.get_last_step_obs()
        img = self.get_obs_img(cur_obs['image'])
        self.imgs.append(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0))

        ee_pose_world = cur_obs['agent']['ee_pose_world'].reshape(4, 4)
        base_pose_world = cur_obs['agent']['arm_base_pose_world'].reshape(4, 4)
        cur_ee_pose = np.matmul(np.linalg.inv(base_pose_world), ee_pose_world)
        self.robot_pos.append(torch.from_numpy(cur_ee_pose[:, 3]).to(torch.float32).unsqueeze(0))
        self.robot_pos[-1][0, 3] = 0
        self.robot_orn.append(torch.from_numpy(R.from_matrix(cur_ee_pose[:3, :3]).as_quat()).to(torch.float32).unsqueeze(0))

        if len(self.imgs) == 1:
            self.imgs *= self.time_sequence_length
            self.robot_pos *= self.time_sequence_length
            self.robot_orn *= self.time_sequence_length
        else:
            self.imgs.pop(0)
            self.robot_pos.pop(0)
            self.robot_orn.pop(0)

    def get_obs_img(self, all_imgs):
        imgs = []
        for cam_view in self.cam_views:
            cam, tp = cam_view.split('_')
            # For now, only RGB image supported
            img = all_imgs[cam + '_camera']['Color']
            img = cv2.resize(img, (320, 256))
            imgs.append(img)
        return np.concatenate(imgs, axis=0)

    def write_results(self, fn, episode_succ):
        with open(fn, "a") as f:
            f.write(''.join([str(int(x)) for x in episode_succ]) + "\n")
        f.close()


@hydra.main('config', 'mt_evaluate', '1.3')
def main(conf: OmegaConf):
    task = hydra.utils.instantiate(conf.task)
    evaluator = MtEvaluator(task, conf.ckpt_dir)
    test_loss(
        ckpt_dir=conf.ckpt_dir,
        evaluator=evaluator,
        cam_views=evaluator.args['cam_view'],
        model_name=conf.model_name,
        proc_name=0,
        epoch=conf.model_name.split('-')[0],
        gpu_name='0',
        using_pos_and_orn=False,
        num_threads=0,
        test_num=conf.test_num
    )


if __name__ == "__main__":
    main()
