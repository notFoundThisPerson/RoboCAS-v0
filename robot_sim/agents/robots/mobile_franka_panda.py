import numpy as np
import sapien.core as sapien
import trimesh
from sapien.core import Pose
from transforms3d.euler import euler2quat, quat2euler
from typing import Union, Optional
import mplib
from scipy.spatial import KDTree

from robot_sim.agents.base_agent import MtBaseAgent
from robot_sim.agents.configs.mobile_franka_panda import defaults
from robot_sim.envs.env_utils import get_point_cloud_from_meshes
from mani_skill2.agents.robots.mobile_panda import get_entities_by_names
from mani_skill2.utils.trimesh_utils import get_articulation_meshes, merge_meshes


class DummyMobileAgent(MtBaseAgent):
    def __init__(self, scene, control_freq, control_mode=None, fix_root_link=True, config=None):
        if control_mode is None:  # if user did not specify a control_mode
            control_mode = "base_pd_pos_arm_pd_ee_abs_pose"
        super().__init__(
            scene,
            control_freq,
            control_mode=control_mode,
            fix_root_link=fix_root_link,
            config=config,
        )

    def _after_init(self):
        # Sanity check
        active_joints = self.robot.get_active_joints()
        assert active_joints[0].name == "agv_base_slider_x"
        assert active_joints[1].name == "agv_base_slider_y"
        assert active_joints[2].name == "agv_base_rotate_z"

        for joint in active_joints:
            if 'finger' in joint.name:
                joint.set_drive_property(100, 100, 20)
                joint.set_friction(0.6)

        # Dummy base
        self.base_link = get_entities_by_names(self.robot.get_links(), "agv_base")

        # Ignore collision between the adjustable body and ground
        s = self.base_link.get_collision_shapes()[0]
        gs = s.get_collision_groups()
        gs[2] = gs[2] | 1 << 30
        s.set_collision_groups(*gs)

    def get_proprioception(self):
        state_dict = super().get_proprioception()
        state_dict["drive_target"] = np.array([joint.get_drive_target() for joint in self.robot.get_active_joints()])
        qpos, qvel = state_dict["qpos"], state_dict["qvel"]
        base_pos, base_orientation, arm_qpos = qpos[:2], qpos[2], qpos[3:-2]
        base_vel, base_ang_vel, arm_qvel = qvel[:2], qvel[2], qvel[3:-2]

        state_dict["arm_qpos"] = arm_qpos
        state_dict["arm_qvel"] = arm_qvel
        state_dict["base_pos"] = base_pos
        state_dict["base_orientation"] = base_orientation
        state_dict["base_vel"] = base_vel
        state_dict["base_ang_vel"] = base_ang_vel
        state_dict['drive_target'] = self.robot.get_drive_target()
        state_dict["gripper_width"] = sum(qpos[-2:])
        state_dict["gripper_target"] = sum(state_dict["drive_target"][-2:])
        return state_dict

    @property
    def base_pose_world(self):
        return self.base_link.pose

    @property
    def base_pose_root(self):
        return self.robot.pose.inv() * self.base_link.pose

    def set_base_pose_root(self, xy, ori):
        qpos = self.robot.get_qpos()
        qpos[0:2] = xy
        qpos[2] = ori
        self.robot.set_qpos(qpos)


class MobileFrankaPanda(DummyMobileAgent):
    _config: defaults.MobileFrankaPandaDefaultConfig

    @classmethod
    def get_default_config(cls):
        return defaults.MobileFrankaPandaDefaultConfig()

    def _after_init(self):
        super()._after_init()

        self.finger1_joint, self.finger2_joint = get_entities_by_names(
            self.robot.get_joints(),
            ["panda_finger_joint1", "panda_finger_joint2"],
        )
        self.finger1_link, self.finger2_link = get_entities_by_names(
            self.robot.get_links(),
            ["panda_leftfinger", "panda_rightfinger"],
        )
        self.fingertip1_link, self.fingertip2_link = get_entities_by_names(
            self.robot.get_links(),
            ["finger_left_tip", "finger_right_tip"],
        )
        self.hand: sapien.LinkBase = get_entities_by_names(
            self.robot.get_links(), "panda_hand"
        )
        self.ee_link = get_entities_by_names(self.robot.get_links(), self.config.ee_link_name)
        self.arm_base_link = get_entities_by_names(self.robot.get_links(), self.config.arm_base_link_name)

    @property
    def base_pose_world(self):
        return self.arm_base_link.pose

    def get_fingers_info(self):
        fingers_pos = self.get_ee_coords().flatten()
        fingers_vel = self.get_ee_vels().flatten()
        return {
            "fingers_pos": fingers_pos,
            "fingers_vel": fingers_vel,
        }

    def get_ee_coords(self):
        finger_tips = [
            self.finger2_joint.get_global_pose().transform(Pose([0, 0.035, 0])).p,
            self.finger1_joint.get_global_pose().transform(Pose([0, -0.035, 0])).p,
        ]
        return np.array(finger_tips)

    def get_ee_vels(self):
        finger_vels = [
            self.finger2_link.get_velocity(),
            self.finger1_link.get_velocity(),
        ]
        return np.array(finger_vels)

    def get_ee_coords_sample(self):
        l = 0.0355
        r = 0.052
        ret = []
        for i in range(10):
            x = (l * i + (4 - i) * r) / 4
            finger_tips = [
                self.finger2_joint.get_global_pose().transform(Pose([0, x, 0])).p,
                self.finger1_joint.get_global_pose().transform(Pose([0, -x, 0])).p,
            ]
            ret.append(finger_tips)
        return np.array(ret).transpose((1, 0, 2))

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """Build a grasp pose (panda_hand)."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return Pose.from_transformation_matrix(T)

    def reset(self, init_qpos=None):
        if init_qpos is None:
            init_qpos = self.config.init_qpos
        super().reset(init_qpos)

    def _setup_planner(self):
        link_names = [link.get_name() for link in self.robot.get_links()]
        active_joints = self.robot.get_active_joints()
        joint_names = [joint.get_name() for joint in active_joints]
        urdf_file = self.urdf_path
        planner = mplib.Planner(
            urdf=urdf_file,
            srdf=self.urdf_path.replace('.urdf', '.srdf'),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=self.ee_link.get_name(),
            joint_vel_limits=np.ones(len(joint_names) - 2) * 0.5,
            joint_acc_limits=np.ones(len(joint_names) - 2) * 0.5
        )
        return planner

    @property
    def ee_pose_world(self):
        return self.ee_link.pose

    @property
    def ee_pose_root(self):
        return self.robot.get_pose().inv() * self.ee_pose_world

    def compute_ik(self, ee_pose: Union[sapien.Pose, np.ndarray], frame='world', n_init_qpos=20, group='full'):
        planner = self._setup_planner()
        ee_pose = self._preprocess_input_pose(ee_pose, frame)
        qpos = self.robot.get_qpos()
        if group == 'arm':
            mask = [True] * 3 + [False] * (self.robot.dof - 3)
        elif isinstance(group, list):
            assert len(group) == self.robot.dof
            mask = group
        else:
            mask = []
        ik_status, ik_res = planner.IK([*ee_pose.p, *ee_pose.q], qpos, n_init_qpos=n_init_qpos, mask=mask)
        return ik_status, ik_res

    # @staticmethod
    # def steady_path_constraint(planner: mplib.Planner, start_qpos, end_pose, fixed_joint_indices=None,
    #                            thresh=np.deg2rad(60)):
    #     planner.robot.set_qpos(start_qpos[:-2])
    #     ee_idx = planner.link_name_2_idx[planner.move_group]
    #     start_pose = planner.robot.get_pinocchio_model().get_link_pose(ee_idx)
    #     start_rotvec = R.from_quat(start_pose[[4, 5, 6, 3]]).as_rotvec()
    #     if isinstance(end_pose, Pose):
    #         end_rot_vec = R.from_quat(end_pose.q[[1, 2, 3, 0]]).as_rotvec()
    #     elif isinstance(end_pose, np.ndarray):
    #         end_rot_vec = R.from_matrix(end_pose[:3, :3]).as_rotvec()
    #     else:
    #         assert isinstance(end_pose, List)
    #         end_pose = [end_pose[i] for i in (4, 5, 6, 3)]
    #         end_rot_vec = R.from_quat(end_pose).as_rotvec()
    #     direction = end_rot_vec - start_rotvec
    #     direction /= np.linalg.norm(direction)
    #
    #     def constraint_f(qpos, out):
    #         planner.robot.set_qpos(qpos)
    #         cur_pose = planner.robot.get_pinocchio_model().get_link_pose(ee_idx)
    #         cur_rotvec = R.from_quat(cur_pose[[4, 5, 6, 3]]).as_rotvec()
    #         dist = np.linalg.norm(np.cross(cur_rotvec - start_rotvec, direction))
    #         out[0] = np.clip(dist - thresh, 0, np.inf)
    #         # if fixed_joint_indices is not None:
    #         #     for idx in fixed_joint_indices:
    #         #         out[0] += abs(qpos[idx] - start_qpos[idx])
    #
    #     def constraint_j(qpos, out):
    #         if fixed_joint_indices is None:
    #             return
    #         full_qpos = planner.pad_qpos(qpos)
    #         jac = planner.robot.get_pinocchio_model().compute_single_link_jacobian(
    #             full_qpos, len(planner.move_group_joint_indices) - 1)
    #         for i in fixed_joint_indices:
    #             out[0, i] = np.linalg.norm(jac[:, i])
    #
    #     return constraint_f, constraint_j

    def path_planning(self, ee_pose: Union[sapien.Pose, np.ndarray], frame='world', point_cloud: Optional[np.ndarray] = None,
                      attatched_objs=[], group='full'):
        planner = self._setup_planner()
        cur_qpos = self.robot.get_qpos()

        use_point_cloud = point_cloud is not None
        if use_point_cloud:
            robot_mesh: trimesh.Trimesh = merge_meshes(get_articulation_meshes(self.robot))
            robot_cloud = get_point_cloud_from_meshes([robot_mesh], 5e-3)
            robot_tree = KDTree(robot_cloud)
            cloud_tree = KDTree(point_cloud)
            ball_indices = sum(robot_tree.query_ball_tree(cloud_tree, 0.01), [])
            remain_indices = [i for i in range(len(point_cloud)) if i not in ball_indices]
            point_cloud = point_cloud[remain_indices]

            root_pose_mat = self.robot.pose.to_transformation_matrix()
            point_cloud = np.matmul(point_cloud - root_pose_mat[np.newaxis, :3, 3], root_pose_mat[:3, :3])

            dist_to_root = np.linalg.norm(point_cloud[:, :2], axis=1)
            point_cloud = point_cloud[dist_to_root < 1.5]
            planner.update_point_cloud(point_cloud)

        use_attatched_obj = len(attatched_objs) > 0
        if use_attatched_obj:
            for obj in attatched_objs:
                obj_mesh = obj['mesh_file']
                pose = self.ee_link.pose.inv() * obj['pose_in_world']
                planner.update_attached_mesh(obj_mesh, [*pose.p, *pose.q])

        if group == 'base':
            assert frame == 'root'
            target_base_joint = [*ee_pose.p[:2], quat2euler(ee_pose.q, 'sxyz')[2]]
            target_qpos = cur_qpos.copy()
            target_qpos[:3] = target_base_joint
            fixed_joint_indices = planner.move_group_joint_indices[3:]
            result = planner.plan_qpos_to_qpos([target_qpos], cur_qpos, self.scene.get_timestep(),
                                               use_point_cloud=use_point_cloud, use_attach=use_attatched_obj,
                                               fixed_joint_indices=fixed_joint_indices, **self.config.planner_config)
        else:
            if group == 'full':
                mask = []
            else:
                mask = [True] * 3 + [False] * (self.robot.dof - 3)
            check_agv_face_dir = group != 'arm'
            ee_pose = self._preprocess_input_pose(ee_pose, frame)
            result = self._plan_qpos_to_pose(planner, [*ee_pose.p, *ee_pose.q], cur_qpos,
                                             time_step=self.scene.get_timestep(), use_point_cloud=use_point_cloud,
                                             use_attach=use_attatched_obj, check_agv_face_dir=check_agv_face_dir, mask=mask)
        if result['status'] != "Success":
            print(result['status'])
        return result

    def _plan_qpos_to_pose(self, planner, goal_pose, current_qpos, mask=[], time_step=0.1, use_point_cloud=False,
                           use_attach=False, constraint_function=None, constraint_jacobian=None, check_agv_face_dir=True):
        ik_status, goal_qpos = planner.IK(goal_pose, current_qpos, mask)
        if ik_status != "Success":
            return {"status": ik_status}
        goal_qpos = sorted(goal_qpos, key=lambda qpos: np.sum(np.abs(qpos - current_qpos)))

        if check_agv_face_dir:
            for qpos in goal_qpos:
                while qpos[2] > np.pi:
                    qpos[2] -= 2 * np.pi
                while qpos[2] < -np.pi:
                    qpos[2] += 2 * np.pi
                vec = goal_pose[:2] - qpos[:2]
                theta = np.arctan2(vec[1], vec[0])
                diff = theta - qpos[2]
                new_panda_joint1 = np.clip(qpos[3] - diff, *self.robot.get_active_joints()[3].get_limits()[0])
                new_rotate_z = qpos[2] + qpos[3] - new_panda_joint1
                qpos[2] = new_rotate_z
                qpos[3] = new_panda_joint1

        if self.config.planner_config['verbose']:
            print("IK results:")
            for i in range(len(goal_qpos)):
                print(goal_qpos[i])

        fixed_joint_indices = [i for i, x in enumerate(mask) if x]
        # const_f, const_j = self.steady_path_constraint(planner, current_qpos, goal_pose, fixed_joint_indices)
        # return planner.plan_qpos_to_qpos(goal_qpos, current_qpos, time_step, use_point_cloud=use_point_cloud,
        #                                  use_attach=use_attach, constraint_function=const_f,
        #                                  constraint_jacobian=const_j, **self.config.planner_config)
        return planner.plan_qpos_to_qpos(goal_qpos, current_qpos, time_step, use_point_cloud=use_point_cloud,
                                         use_attach=use_attach, constraint_function=constraint_function,
                                         constraint_jacobian=constraint_jacobian, fixed_joint_indices=fixed_joint_indices,
                                         **self.config.planner_config)

    def _preprocess_input_pose(self, ee_pose: Union[sapien.Pose, np.ndarray], frame='world'):
        if isinstance(ee_pose, np.ndarray):
            ee_pose = sapien.Pose.from_transformation_matrix(ee_pose)
        if frame == 'world':
            ee_pose = self.robot.pose.inv() * ee_pose
        elif frame == 'base':
            ee_pose = self.robot.pose.inv() * self.base_pose_world * ee_pose
        else:
            assert frame == 'root'
        ee_pose.set_q(ee_pose.q / np.linalg.norm(ee_pose.q))
        return ee_pose

    def get_proprioception(self):
        state_dict = super().get_proprioception()
        state_dict['ee_pose_world'] = self.ee_pose_world.to_transformation_matrix().reshape(-1)
        state_dict['arm_base_pose_world'] = self.arm_base_link.pose.to_transformation_matrix().reshape(-1)
        return state_dict

    @property
    def gripper_open_width(self):
        return self.robot.get_active_joints()[-1].get_limits()[0, 1]

    @property
    def gripper_close_width(self):
        return self.robot.get_active_joints()[-1].get_limits()[0, 0]


if __name__ == '__main__':
    from robot_sim.utils import create_default_world

    _, scene, viewer = create_default_world()
    robot = MobileFrankaPanda(scene, 30)
    robot.reset()
    ee_link = get_entities_by_names(robot.robot.get_links(), robot.config.ee_link_name)

    cnt = 0
    while not viewer.closed:
        if cnt % 500 < 250:
            act = {'base': [-0.1, 0, np.pi / 4], 'arm': [0.3, 0.4, 0.5, np.pi, 0, 0], 'gripper': 0}
        else:
            act = {'base': [0, 0.1, 0], 'arm': [0.3, -0.4, 0.5, np.pi, 0, 0], 'gripper': 1}
        act = robot.controller.from_action_dict(act)
        robot.set_action(act)
        for _ in range(round(1 / scene.get_timestep() / robot._control_freq)):
            robot.before_simulation_step()
            scene.step()
        print(cnt, ee_link.pose)
        scene.update_render()
        viewer.render()
        cnt += 1
