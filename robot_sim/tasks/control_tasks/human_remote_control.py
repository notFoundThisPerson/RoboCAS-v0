import hydra
from omegaconf import OmegaConf
import os
import numpy as np
import cv2

from mani_skill2.utils.visualization.cv2_utils import OpenCVViewer
from robot_sim.tasks.control_tasks.base_control_task import BaseControlTask


def get_image_from_obs(obs: dict):
    global frame_cache
    try:
        frame = []
        for cam_imgs in obs['image'].values():
            frame.append(cam_imgs['Color'])
        frame_cache = np.concatenate(frame, 1)
    except (TypeError, KeyError):
        pass
    return frame_cache


@hydra.main(os.path.join(os.path.dirname(__file__), '../../config'), 'human_remote_control', '1.3')
def main(conf: OmegaConf):
    task: BaseControlTask = hydra.utils.instantiate(conf.task)
    # task.enable_logging = False
    obs = task.reset()
    if task.env.render_mode == 'human':
        task.env.render_human()
    global step_cnt
    step_cnt = 0

    def render_wait():
        while True:
            task.env.render_human()
            sapien_viewer = task.env.viewer
            if sapien_viewer.window.key_down("0"):
                break

    def key2motion(key):
        xyz, rot = [0, 0, 0], [0, 0, 0]
        change_gripper = False
        rot_3_deg = 0.05235987755982989
        if key == 'q':
            xyz[2] = 0.01
        elif key == 'e':
            xyz[2] = -0.01
        elif key == 'a':
            xyz[0] = 0.01
        elif key == 'd':
            xyz[0] = -0.01
        elif key == 'w':
            xyz[1] = 0.01
        elif key == 's':
            xyz[1] = -0.01

        elif key == '8':
            rot[0] = -rot_3_deg
        elif key == '5':
            rot[0] = rot_3_deg
        elif key == '4':
            rot[2] = -rot_3_deg
        elif key == '6':
            rot[2] = rot_3_deg
        elif key == '7':
            rot[1] = rot_3_deg
        elif key == '9':
            rot[1] = -rot_3_deg

        elif key == 'g':
            change_gripper = True

        elif key == '0':
            render_wait()
        elif key == 'r':
            global step_cnt
            step_cnt = 0
            task.reset()
        elif key == 'p':
            task.record_episode_obs()

        return xyz, rot, change_gripper

    opencv_viewer = OpenCVViewer(exit_on_esc=False)
    gripper_cmd = 1
    while True:
        step_cnt += 1
        frame = get_image_from_obs(obs)
        cv2.putText(frame, '%d' % step_cnt, (30, frame.shape[0] - 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 2)
        key = opencv_viewer.imshow(frame)
        xyz, rot, change_gripper = key2motion(key)
        if change_gripper:
            gripper_cmd = 1 - gripper_cmd
        obs = task.step(xyz, rot, gripper_cmd)


if __name__ == '__main__':
    main()
