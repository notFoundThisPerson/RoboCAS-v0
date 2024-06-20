from pathlib import Path
from typing import Union, Optional
import numpy as np
from time import localtime, strftime
import cv2
from scipy.spatial.transform import Rotation as R
from sapien.core import Pose
from transforms3d.euler import euler2quat

from robot_sim.envs.basic_env import TaskBaseEnv
from robot_sim.utils import DataLog
from robot_sim.tasks.basic_actions.gripper_actions import GripperOpenCloseAction
from robot_sim.tasks.basic_actions.end_effector_action import RelativeEEPoseAction


class BaseControlTask:
    def __init__(self, env: TaskBaseEnv, output_data_path: Optional[Union[Path, str]] = None):
        self.env = env
        self.agent = self.env.agent
        self.set_record_path(output_data_path)
        self.cmd_log = []
        self.exec_log = []
        self.start_time = localtime()
        self.enable_logging = True

    def set_record_path(self, record_path: Optional[Union[Path, str]] = None):
        self.record_path = record_path
        if isinstance(self.record_path, str):
            self.record_path = Path(self.record_path)
        if self.record_path is not None:
            self.record_path.mkdir(parents=True, exist_ok=True)

    def reset(self):
        self.cmd_log.clear()
        self.exec_log.clear()
        obs, _ = self.env.reset()
        self.exec_log.append(DataLog.from_list([obs]))
        self.cmd_log.append(None)
        self.start_time = localtime()
        return obs

    def get_last_step_obs(self):
        if len(self.exec_log) == 0:
            return None
        return self.exec_log[-1].last_log

    def step(self, delta_ee_pos, delta_ee_rot, gripper_cmd, max_steps=10):
        _, step_log = GripperOpenCloseAction('open' if gripper_cmd == 1 else 'close', self.env).run()
        delta_pose = Pose(delta_ee_pos, euler2quat(*delta_ee_rot, 'sxyz'))
        move_task = RelativeEEPoseAction(delta_pose, self.env, 'ee', max_steps=max_steps)
        # move_task.check_success_input['success_fn'] = lambda: True
        _, obs_after_step = move_task.run()
        if len(obs_after_step) == 0:
            print('failed')
        step_log.extend(obs_after_step)
        if self.enable_logging:
            self.cmd_log.append((delta_ee_pos, delta_ee_rot, gripper_cmd))
            self.exec_log.append(step_log)
        if self.env.render_mode == 'human':
            self.env.render()
        return step_log.last_log

    def record_episode_obs(self, used_cams=None, lang_goal=''):
        if self.record_path is None:
            return
        output_vid_path = self.record_path / strftime('%Y.%m.%d-%H.%M.%S.mp4', self.start_time)
        frame_size = (640 * len(self.agent.config.cameras), 480)
        writer = cv2.VideoWriter(output_vid_path.as_posix(), cv2.VideoWriter_fourcc(*'mp4v'), 15, frame_size)
        cnt = 0

        def cmd2str(cmd_info):
            if isinstance(cmd_info, float):
                info_str = '%.3f' % cmd_info
            elif isinstance(cmd_info, (list, tuple)):
                info_str = '('
                for i, data in enumerate(cmd_info):
                    info_str += cmd2str(data)
                    if i != len(cmd_info) - 1:
                        info_str += ', '
                info_str += ')'
            elif isinstance(cmd_info, dict):
                info_str = '{'
                for i, (k, v) in enumerate(cmd_info.items()):
                    info_str += cmd2str(k) + ': ' + cmd2str(v)
                    if i != len(cmd_info) - 1:
                        info_str += ', '
                info_str += '}'
            else:
                info_str = str(cmd_info)
            return info_str

        for step, (obs, cmd) in enumerate(zip(self.exec_log, self.cmd_log)):
            if obs is None:
                continue
            cam_names = list(obs['image'].keys())
            for idx in range(len(obs)):
                frame = np.zeros([frame_size[1], frame_size[0], 3], dtype=np.uint8)
                cnt += 1
                for cam_idx, cam_name in enumerate(cam_names):
                    color = obs['image'][cam_name]['Color'][idx]
                    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                    if used_cams is None or cam_name not in used_cams:
                        text_color = (0, 0, 255)
                    else:
                        text_color = (0, 255, 0)
                    cv2.putText(color, cam_name, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, text_color)
                    frame[:, cam_idx * 640:(cam_idx + 1) * 640] = cv2.resize(color, (640, 480))
                cv2.putText(frame, '%s' % cnt, (10, 470), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))

                obs['agent']['ee_pose_world'] = obs['agent']['ee_pose_world'].reshape(-1, 4, 4)
                xyz = tuple(obs['agent']['ee_pose_world'][idx, :3, 3].tolist())
                quat = tuple(R.from_matrix(obs['agent']['ee_pose_world'][idx, :3, :3]).as_quat().tolist())
                xyz_info = 'xyz=[%.3f, %.3f, %.3f]' % xyz
                quat_info = 'quat=[%.3f, %.3f, %.3f, %.3f]' % quat
                cv2.putText(frame, xyz_info, (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))
                cv2.putText(frame, quat_info, (10, 90), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))
                if len(lang_goal) > 0:
                    cv2.putText(frame, lang_goal, (10, 440), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))

                cmd_info = 'cmd=' + cmd2str(cmd)
                cv2.putText(frame, cmd_info, (10, 150), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))
                writer.write(frame)
        writer.release()
        print('Video result wrote to %s' % output_vid_path)

    # def check_task_succeed(self, **kwargs):
    #     raise NotImplementedError
