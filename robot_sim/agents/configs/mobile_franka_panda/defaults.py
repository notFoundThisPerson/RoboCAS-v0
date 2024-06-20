from copy import deepcopy
import numpy as np
from transforms3d.euler import euler2quat

from robot_sim.agents.controllers import *
from mani_skill2.agents.controllers import *
from mani_skill2.sensors.camera import CameraConfig


class MobileFrankaPandaDefaultConfig:
    def __init__(self) -> None:
        self.urdf_path = "{MT_ASSET_DIR}/urdf/mobile_franka_panda.urdf"
        self.urdf_config = dict(
            _materials=dict(
                gripper=dict(static_friction=0.99, dynamic_friction=0.99, restitution=0.0)
            ),
            link=dict(
                finger_left_tip=dict(material="gripper"),
                finger_right_tip=dict(material="gripper"),
                panda_rightfinger=dict(material="gripper"),
                panda_leftfinger=dict(material="gripper"),
            ),
        )

        self.planner_config = dict(
            rrt_range=0.1,
            planning_time=2,
            fix_joint_limits=True,
            verbose=False,
            planner_name="RRTConnect",
            # no_simplification=False,
            constraint_tolerance=1e-3
        )

        self.base_joint_names = [
            "agv_base_slider_x",
            "agv_base_slider_y",
            "agv_base_rotate_z"
        ]

        self.arm_joint_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
        self.base_stiffness = 1e8
        self.base_damping = 1e5
        self.base_force_limit = 1e5
        self.arm_stiffness = 1e8
        self.arm_damping = 1e4
        self.arm_force_limit = 100
        self.arm_joint_delta = 0.1
        self.arm_ee_delta = 0.1
        self.arm_abs_pos_lower = [-1, -1, 0]
        self.arm_abs_pos_upper = [1, 1, 1.5]

        self.gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e3
        self.gripper_force_limit = 100

        self.ee_link_name = "panda_hand_tcp"
        self.arm_base_link_name = "panda_link0"

        self.init_qpos = (0, 0, 0, 0, -1.02, 0, -2.51, 0, 1.35, 0.78, 0.04, 0.04)
        self.camera_h = 1.5

    @property
    def controllers(self):
        basic_controllers = {}  # controller configs for each component

        # -------------------------------------------------------------------------- #
        # Mobile Base
        # -------------------------------------------------------------------------- #
        basic_controllers["base"] = dict(
            # PD ego-centric joint velocity
            base_pd_joint_vel=PDBaseVelControllerConfig(
                self.base_joint_names,
                lower=[-0.5, -0.5, -3.14],
                upper=[0.5, 0.5, 3.14],
                damping=1000,
                force_limit=500,
            ),
            base_pd_pos=PDJointPosControllerConfig(
                self.base_joint_names,
                None,
                None,
                stiffness=self.base_stiffness,
                damping=self.base_damping,
                force_limit=self.base_force_limit,
                normalize_action=False,
            ),
            base_pd_delta_pos=PDJointPosControllerConfig(
                self.base_joint_names,
                lower=[-0.1, -0.1, -0.1],
                upper=[0.1, 0.1, 0.1],
                stiffness=self.base_stiffness,
                damping=self.base_damping,
                force_limit=self.base_force_limit,
                use_delta=True,
                normalize_action=False,
            )
        )

        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        # PD joint velocity
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            lower=-3.14,
            upper=3.14,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
        )

        # PD joint position
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -self.arm_joint_delta,
            self.arm_joint_delta,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee pose
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            self.arm_joint_names,
            -self.arm_ee_delta,
            self.arm_ee_delta,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            self.arm_joint_names,
            -self.arm_ee_delta,  # not used
            self.arm_ee_delta,  # not used
            rot_bound=self.arm_ee_delta,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            normalize_action=False
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True

        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # absolute ee position control（robot root frame）
        arm_pd_ee_abs_pos = PDEEPosControllerConfig(
            self.arm_joint_names,
            self.arm_abs_pos_lower,
            self.arm_abs_pos_upper,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            normalize_action=False,
            use_delta=False,
            frame='base',
        )

        # 末端绝对位姿控制（机器人root frame）
        arm_pd_ee_abs_pose_wrt_robot_base = PDEEPoseControllerConfig(
            self.arm_joint_names,
            [-100, -100, 0],
            [100, 100, 1.5],
            rot_bound=np.pi,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            normalize_action=False,
            use_delta=False,
            interpolate=False,
            frame='base',
        )

        # absolute ee pose control（robot root frame, linear trajectory）
        arm_pd_ee_abs_pose_linear_wrt_robot_base = PDEEPoseLinearControllerConfig(
            self.arm_joint_names,
            [-100, -100, 0],
            [100, 100, 1.5],
            rot_bound=np.pi,
            linear_vel_thresh=0.05,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            normalize_action=False,
            use_delta=False,
            interpolate=False,
            frame='base',
        )

        # absolute ee pose control（robot root frame, circular trajectory）
        arm_pd_ee_abs_pose_circular_wrt_world = PDEEPoseCircularControllerConfig(
            self.arm_joint_names,
            [-100, -100, 0],
            [100, 100, 1.5],
            rot_bound=np.pi,
            linear_vel_thresh=0.05,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            normalize_action=False,
            use_delta=False,
            interpolate=False,
            frame='world',
        )

        arm_pd_ee_abs_pose_wrt_arm_base = PDEEPoseMobileControllerConfig(
            self.arm_joint_names,
            self.arm_abs_pos_lower,
            self.arm_abs_pos_upper,
            rot_bound=np.pi,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            interpolate=False,
        )

        # PD joint position and velocity
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -self.arm_joint_delta,
            self.arm_joint_delta,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )

        basic_controllers["arm"] = dict(
            arm_pd_joint_vel=arm_pd_joint_vel,
            arm_pd_joint_delta_pos=arm_pd_joint_delta_pos,
            arm_pd_joint_pos=arm_pd_joint_pos,
            arm_pd_joint_target_delta_pos=arm_pd_joint_target_delta_pos,
            arm_pd_ee_delta_pos=arm_pd_ee_delta_pos,
            arm_pd_ee_delta_pose=arm_pd_ee_delta_pose,
            arm_pd_ee_target_delta_pos=arm_pd_ee_target_delta_pos,
            arm_pd_ee_target_delta_pose=arm_pd_ee_target_delta_pose,
            arm_pd_joint_pos_vel=arm_pd_joint_pos_vel,
            arm_pd_joint_delta_pos_vel=arm_pd_joint_delta_pos_vel,
            arm_pd_ee_abs_pos=arm_pd_ee_abs_pos,
            arm_pd_ee_abs_pose_wrt_robot_base=arm_pd_ee_abs_pose_wrt_robot_base,
            arm_pd_ee_abs_pose_wrt_arm_base=arm_pd_ee_abs_pose_wrt_arm_base,
            arm_pd_ee_abs_pose_linear_wrt_robot_base=arm_pd_ee_abs_pose_linear_wrt_robot_base,
            arm_pd_ee_abs_pose_circular_wrt_world=arm_pd_ee_abs_pose_circular_wrt_world
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        basic_controllers["gripper"] = dict(
            gripper_pd_joint_pos=PDJointPosMimicControllerConfig(
                self.gripper_joint_names,
                -0.01,  # a trick to have force when the object is thin
                0.04,
                self.gripper_stiffness,
                self.gripper_damping,
                self.gripper_force_limit,
                normalize_action=False
            ),
        )

        controller_configs = {}
        for base_controller_name in basic_controllers["base"]:
            for arm_controller_name in basic_controllers["arm"]:
                c = {"base": basic_controllers["base"][base_controller_name],
                     "arm": basic_controllers["arm"][arm_controller_name],
                     "gripper": basic_controllers["gripper"]["gripper_pd_joint_pos"]}
                combined_name = base_controller_name + "_" + arm_controller_name
                controller_configs[combined_name] = c

        global_controllers = dict(
            global_pd_ee_abs_pose_linear_wrt_robot_base=PDEEPoseLinearControllerConfig(
                self.base_joint_names + self.arm_joint_names,
                [-100, -100, 0],
                [100, 100, 1.5],
                rot_bound=np.pi,
                linear_vel_thresh=0.1,
                stiffness=self.arm_stiffness,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
                ee_link=self.ee_link_name,
                normalize_action=False,
                use_delta=False,
                interpolate=False,
                frame='base'
            ),
            global_pd_ee_abs_pose_circular_wrt_world=PDEEPoseCircularControllerConfig(
                self.base_joint_names + self.arm_joint_names,
                [-100, -100, 0],
                [100, 100, 1.5],
                rot_bound=np.pi,
                linear_vel_thresh=0.1,
                stiffness=self.arm_stiffness,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
                ee_link=self.ee_link_name,
                normalize_action=False,
                use_delta=False,
                interpolate=False,
                frame='world'
            )
        )
        for controller_name, controller in global_controllers.items():
            c = {"global": controller,
                 "gripper": basic_controllers["gripper"]["gripper_pd_joint_pos"]}
            controller_configs[controller_name] = c

        controller_configs['base_follow_joint_arm_follow_joint'] = CombinedSequenceControllerConfig()
        controller_configs['base_follow_joint_arm_follow_joint'].configs = dict(
            base=PDFollowJointControllerConfig(
                self.base_joint_names,
                None,
                None,
                stiffness=self.arm_stiffness,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
                use_delta=False,
            ),
            arm=PDFollowJointControllerConfig(
                self.arm_joint_names,
                None,
                None,
                stiffness=self.arm_stiffness,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
                use_delta=False,
            ),
            gripper=basic_controllers["gripper"]["gripper_pd_joint_pos"]
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    @property
    def cameras(self):
        cameras = [
            CameraConfig(
                'base_camera',
                p=[0, 0, 0],
                q=[1, 0, 0, 0],
                width=640,
                height=480,
                near=0.005,
                far=5,
                fov=np.pi / 2,
                actor_uid='base_camera_link'
            ),
            CameraConfig(
                'gripper_camera',
                p=[0, 0, 0],
                q=[1, 0, 0, 0],
                width=640,
                height=480,
                near=0.001,
                far=5,
                fov=np.pi / 2,
                actor_uid='gripper_camera_link',
                texture_names=("Color", "Position", "Segmentation")
            ),
            CameraConfig(
                'static_camera',
                p=[1, 1.5, 1.5],
                q=euler2quat(*np.deg2rad([0, 60, -90]), 'sxyz'),
                width=640,
                height=480,
                near=0.05,
                far=10,
                fov=np.pi / 2
            )
        ]
        return cameras
