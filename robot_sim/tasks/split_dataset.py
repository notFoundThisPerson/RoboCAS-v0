import json
import os
import argparse
from random import shuffle
from scipy.spatial.transform import Rotation
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, help='Path to the dataset')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    episode_idx = [int(f[-7:]) for f in os.listdir(args.data_root) if f.startswith('episode_')]
    shuffle(episode_idx)
    num_train = int(len(episode_idx) * 0.9)
    info = dict(
        train_eps=episode_idx[:num_train],
        val_eps=episode_idx[num_train:],
        ep_length={},
        step_goals={},
        language_goal={},
    )
    pos_act_space = []
    rot_act_space = []

    for i in sorted(episode_idx):
        epi_info_path = os.path.join(args.data_root, 'episode_%07d/episode_info.npz' % i)
        epi_info = np.load(epi_info_path, allow_pickle=True)
        epi_length = epi_info['episode_length'].item()
        info['language_goal'][i] = epi_info['language_goal'].item()
        info['ep_length'][i] = epi_length

        info['step_goals'][i] = []
        last_goal = '******'
        step_goal_type = epi_info['step_goal_type'].tolist()
        for step, goal_type in enumerate(step_goal_type):
            if goal_type != last_goal:
                info['step_goals'][i].append({'goal_type': goal_type,
                                              'goal_range': [step, step],
                                              'goal_description': epi_info['step_lang_goals'][step].item()})
                last_goal = goal_type
            else:
                info['step_goals'][i][-1]['goal_range'][1] = step
        pos_act_space.append(epi_info['rel_pos'])
        rot_act_space.append(epi_info['rel_orn'])

    pos_act_space = np.concatenate(pos_act_space, 0)
    rot_act_space = np.concatenate(rot_act_space, 0)
    rot_act_space = Rotation.from_quat(rot_act_space).as_euler('xyz', False)

    info['action_space'] = dict(
        relative_position=(np.percentile(pos_act_space, 1, axis=0).tolist(),
                           np.percentile(pos_act_space, 99, axis=0).tolist()),
        relative_rotation=(np.percentile(rot_act_space, 1, axis=0).tolist(),
                           np.percentile(rot_act_space, 99, axis=0).tolist())
    )

    with open(os.path.join(args.data_root, 'data_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
