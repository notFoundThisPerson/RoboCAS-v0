from random import sample, randint
from transforms3d.quaternions import mat2quat
from transforms3d.euler import euler2quat, euler2mat
import numpy as np
from typing import Type, Optional, List, Dict
import sapien.core as sapien
import hydra
from omegaconf import OmegaConf
import logging

from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.sapien_utils import get_entity_by_name, set_articulation_render_material
from mani_skill2.utils.trimesh_utils import merge_meshes
from robot_sim.agents.base_agent import MtBaseAgent
from robot_sim.agents.robots.mobile_franka_panda import MobileFrankaPanda
from robot_sim.envs.objects import GraspNetObject, ArticulationObject
from robot_sim.envs.objects.create_random_room_bkg import gen_room_obj
from robot_sim.envs.env_utils import get_point_cloud_from_meshes
from robot_sim.utils import format_path


class TaskBaseEnv(BaseEnv):
    SUPPORTED_ROBOTS = {'mobile_franka_panda': MobileFrankaPanda}

    def __init__(
            self,
            render_mode='human',
            control_mode='base_pd_pos_arm_pd_ee_abs_pose_wrt_arm_base',
            num_objects=5,
            sim_freq=240,
            control_freq=10,
            obj_place_mode='random',
            rand_rotate_obj_around_z=True,
            rand_rotate_obj_xy=False,
            tight_factor=1.0,
            obj_repeat_num=1,
            articulation_configs: Optional[OmegaConf] = None,
            use_room=True,
            random_background=False,
            actor_configs=None,
            **kwargs
    ):
        '''
        :param kwargs: Other environment configuration options, see details in the __init__ method of BaseEnv
            Additional supported keywords:
                env_config: dict    Other configurations for objects in the current environment
                robot_config: dict  Other configurations for the robot in the current environment
        '''
        self.robot_uid = 'mobile_franka_panda'
        self.num_objects = num_objects
        self.obj_place_mode = obj_place_mode
        self.rand_rotate_obj_around_z = rand_rotate_obj_around_z
        self.rand_rotate_obj_xy = rand_rotate_obj_xy
        self.tight_factor = tight_factor
        self.use_room = use_room
        self.random_background = random_background
        self.obj_repeat_num = obj_repeat_num

        self._articulation_configs = articulation_configs
        self._robot_custom_config = kwargs.pop('robot_config', dict())
        self.objs: List[GraspNetObject] = []
        self.articulations: List[ArticulationObject] = []
        self.last_env_state, self.pcd_cache = None, None
        self.orig_obj_num = 0
        self.actor_configs = actor_configs if actor_configs is not None else dict()
        super().__init__(render_mode=render_mode, control_mode=control_mode, sim_freq=sim_freq, control_freq=control_freq,
                         **kwargs)

    def set_main_rng(self, seed):
        self._main_rng = np.random.RandomState(np.random.RandomState().randint(2**32))

    def _load_articulations(self):
        self.ground = self._add_ground(render=self.bg_name is None, room=self.use_room, random_texture=self.random_background)
        if self._articulation_configs is None:
            return
        for config in self._articulation_configs.values():
            self.articulations.append(hydra.utils.instantiate(config, scene=self.scene))

    def _clear(self):
        super()._clear()
        self.articulations.clear()
        self.objs.clear()

    def _configure_agent(self):
        agent_cls: Type[MtBaseAgent] = self.SUPPORTED_ROBOTS[self.robot_uid]
        self._agent_cfg = self._robot_custom_config.pop('default_config', agent_cls.get_default_config())

        for k, v in self._robot_custom_config.items():
            if hasattr(self._agent_cfg, k):
                setattr(self._agent_cfg, k, v)

    def _initialize_agent(self):
        self.agent.reset()
        base_pose = self._robot_custom_config.get('base_pose', None)
        if base_pose is None:
            base_pose = sapien.Pose()
        else:
            base_pose = dict(base_pose)
            if 'q' in base_pose and len(base_pose['q']) == 3:
                base_pose['q'] = np.deg2rad(base_pose['q'])
                base_pose['q'] = euler2quat(*base_pose['q'], 'sxyz')
            base_pose = sapien.Pose(p=base_pose['p'], q=base_pose['q'])
        self.agent.set_root_pose(base_pose)

    def _load_agent(self):
        agent_cls: Type[MtBaseAgent] = self.SUPPORTED_ROBOTS[self.robot_uid]
        self.agent: MtBaseAgent = agent_cls(self._scene, self._control_freq, self._control_mode, config=self._agent_cfg)
        self.tcp: sapien.Link = get_entity_by_name(
            self.agent.robot.get_links(), self.agent.config.ee_link_name
        )
        set_articulation_render_material(self.agent.robot, specular=0.9, roughness=0.3)

    def _initialize_articulations(self):
        for art in self.articulations:
            art.set_pose(art.config.initial_pose)
            if art.config.initial_joint_status is not None:
                qpos = []
                for joint in art.model.get_active_joints():
                    qpos.append(art.config.initial_joint_status.get(joint.name, joint.get_limits()[0][0]))
                art.model.set_qpos(qpos)
                art.model.set_drive_target(qpos)

    def _initialize_actors(self):
        height_margin = 0.01
        # for tight arrange
        last_surf_idx = 0
        last_y_min = None

        to_remove_list = []

        def remove_object_from_scene(obj: GraspNetObject, index):
            self._scene.remove_actor(obj.model)
            to_remove_list.append(index)

        surfaces = []
        articulations_with_surfs = []
        repeated_objs = []

        def fix_obj_height(pose, z_lb):
            # Avoid collision between table and object
            positon = pose.p
            positon[2] -= z_lb - height_margin
            pose.set_p(positon)
            return pose

        for art in self.articulations:
            if len(art.surfaces) > 0:
                surfaces.extend(list(art.surfaces.values()))
                articulations_with_surfs.extend([art] * len(art.surfaces))

        tmp_bucket = []
        obj_surf_idx = []
        if self.obj_place_mode == 'pile':
            # a virtual bucket, to prevent object from flying around
            for i, (art, surf) in enumerate(zip(articulations_with_surfs, surfaces)):
                builder = self.scene.create_actor_builder()
                x_size_half = (surf[1][0] - surf[0][0]) / 2
                y_size_half = (surf[1][1] - surf[0][1]) / 2
                material = self.scene.create_physical_material(0, 0, 0.5)
                builder.add_box_collision(sapien.Pose([-x_size_half, 0, 0.5]), [1e-4, y_size_half, 0.5], material)
                builder.add_box_collision(sapien.Pose([x_size_half, 0, 0.5]), [1e-4, y_size_half, 0.5], material)
                builder.add_box_collision(sapien.Pose([0, -y_size_half, 0.5]), [x_size_half, 1e-4, 0.5], material)
                builder.add_box_collision(sapien.Pose([0, y_size_half, 0.5]), [x_size_half, 1e-4, 0.5], material)
                builder.add_box_collision(sapien.Pose([0, 0, 0]), [x_size_half, y_size_half, 1e-4], material)
                builder.add_box_collision(sapien.Pose([0, 0, 1]), [x_size_half, y_size_half, 1e-4], material)
                wall = builder.build_static()
                wall.set_pose(sapien.Pose([1, -1, i]))
                tmp_bucket.append(wall)

        for i, obj in enumerate(self.objs):
            if i >= self.orig_obj_num:
                remove_object_from_scene(obj, i)
                continue
            if len(surfaces) == 0:
                pose = sapien.Pose(p=[np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0],
                                   q=euler2quat(0, 0, np.random.uniform(-np.pi, np.pi), 'sxyz'))
                lb, _ = obj.get_AABB(local_frame=True)
                pose = fix_obj_height(pose, lb[2])
                obj.model.set_pose(pose)
            else:
                if last_surf_idx >= len(surfaces):
                    logging.info('There is no place remaining for object "%s", remove it from the scene' % obj.name)
                    remove_object_from_scene(obj, i)
                    continue
                local_transform = np.eye(4)
                if self.rand_rotate_obj_around_z:
                    angle = np.random.uniform(-np.pi, np.pi)
                    local_transform[:3, :3] = euler2mat(0, 0, angle, 'sxyz')
                if self.rand_rotate_obj_xy and np.random.rand() > 0.4:
                    angle = np.random.uniform([-np.pi, -np.pi, 0], [np.pi, np.pi, 0])
                    local_transform[:3, :3] = np.matmul(local_transform[:3, :3], euler2mat(*angle, 'sxyz'))
                lb, ub = obj.get_AABB(local_frame=True, local_transform=local_transform)

                if self.obj_place_mode == 'tight':
                    sizes = ub - lb

                    layer_lb, layer_ub = surfaces[last_surf_idx]
                    if last_y_min is None:
                        last_y_min = layer_lb[1]
                    elif last_y_min + sizes[1] > layer_ub[1]:
                        last_surf_idx += 1
                        if last_surf_idx >= len(surfaces):
                            remove_object_from_scene(obj, i)
                            continue
                        layer_lb, layer_ub = surfaces[last_surf_idx]
                        last_y_min = layer_lb[1]
                    for j in range(self.obj_repeat_num):
                        place_p = [layer_ub[0] - ub[0] - sizes[0] * j * self.tight_factor, last_y_min - lb[1], layer_lb[2]]
                        if place_p[0] + lb[0] < layer_lb[0]:
                            break
                        pose = sapien.Pose(place_p, mat2quat(local_transform[:3, :3]))
                        pose = articulations_with_surfs[last_surf_idx].pose * pose
                        pose = fix_obj_height(pose, lb[2])
                        if j == 0:
                            obj.model.set_pose(pose)
                        else:
                            new_obj = obj.__class__(self.scene, obj.dataset_path.as_posix(), obj.obj_index)
                            new_obj.model.set_pose(pose)
                            repeated_objs.append(new_obj)
                    last_y_min += sizes[1] * self.tight_factor
                elif self.obj_place_mode == 'random':
                    art = sample(set(articulations_with_surfs), k=1)[0]
                    pose = art.sample_place_position() * sapien.Pose.from_transformation_matrix(local_transform)
                    pose = fix_obj_height(pose, lb[2])
                    obj.model.set_pose(pose)
                elif self.obj_place_mode == 'pile':
                    surf_idx = randint(0, len(surfaces) - 1)
                    obj_surf_idx.append(surf_idx)
                    bucket = tmp_bucket[surf_idx]
                    drop_pose = bucket.pose * sapien.Pose([0, 0, 0.3]) * sapien.Pose.from_transformation_matrix(local_transform)
                    obj.model.set_pose(drop_pose)
                    for _ in range(self.sim_freq):
                        self.scene.step()
                else:
                    raise NotImplementedError

        if self.obj_place_mode == 'pile':
            for obj, surf_idx in zip(self.objs, obj_surf_idx):
                surf = surfaces[surf_idx]
                bucket = tmp_bucket[surf_idx]
                new_pose = (articulations_with_surfs[surf_idx].pose *
                            sapien.Pose([(surf[0][0] + surf[1][0]) / 2, (surf[0][1] + surf[1][1]) / 2, surf[1][2]]) *
                            bucket.pose.inv() * obj.model.pose)
                obj.model.set_pose(new_pose)
            for obj in tmp_bucket:
                self.scene.remove_actor(obj)

        for idx in reversed(to_remove_list):
            obj = self.objs.pop(idx)
            if isinstance(obj.model, sapien.Articulation):
                self.scene.remove_articulation(obj.model)
            else:
                self.scene.remove_actor(obj.model)
        self.orig_obj_num = len(self.objs)
        self.objs.extend(repeated_objs)

        for _ in range(self.sim_freq):
            self.scene.step()

        for obj in self.objs:
            obj.model.set_velocity([0, 0, 0])
            obj.model.set_angular_velocity([0, 0, 0])

    def reset(self, seed=None, options=None):
        if options is None or not options.get('reconfigure', False):
            for obj in self.objs:
                self.scene.remove_actor(obj.model)
            self.objs.clear()
            self._load_actors()
        return super().reset(None, options)

    @property
    def cameras(self):
        return self._cameras

    def initialize_episode(self):
        self._initialize_agent()
        self._initialize_articulations()
        self._initialize_actors()
        self._initialize_task()

    def _load_actors(self):
        self.objs = []
        obj_list = self.actor_configs.get('obj_list', list(range(68)))
        target_obj_indices = np.random.choice(obj_list, self.num_objects, replace=self.actor_configs.get('allow_repeat', False))
        self.orig_obj_num = self.num_objects
        for obj_idx in target_obj_indices:
            self.objs.append(GraspNetObject(self._scene, '{MT_ASSET_DIR}/graspnet', str(obj_idx)))

    def _add_ground(self, altitude=0.0, render=True, room=True, random_texture=False):
        if not render:
            return self._scene.add_ground(altitude=altitude, render=False)
        if not room:
            rend_mtl = self._renderer.create_material()
            rend_mtl.metallic = 0.0
            rend_mtl.roughness = 0.9
            rend_mtl.specular = 0.8
            rend_mtl.base_color = [0.06, 0.08, 0.12, 1]
            return self._scene.add_ground(altitude=altitude, render=True, render_material=rend_mtl)

        room_size = [6, 6, 4]
        gen_room_obj(*room_size, random_texture)
        half_size = [x / 2. for x in room_size]
        builder = self._scene.create_actor_builder()
        builder.add_box_collision(pose=sapien.Pose([0, 0, 5e-4]), half_size=[20, 20, 1e-3])
        builder.add_box_collision(pose=sapien.Pose([-half_size[0], 0, half_size[2]]),
                                  half_size=[1e-3, room_size[1], room_size[2]])
        builder.add_box_collision(pose=sapien.Pose([half_size[0], 0, half_size[2]]),
                                  half_size=[1e-3, room_size[1], room_size[2]])
        builder.add_box_collision(pose=sapien.Pose([0, -half_size[1], half_size[2]]),
                                  half_size=[room_size[0], 1e-3, room_size[2]])
        builder.add_box_collision(pose=sapien.Pose([0, half_size[1], half_size[2]]),
                                  half_size=[room_size[0], 1e-3, room_size[2]])
        builder.add_visual_from_file(format_path('{MT_ASSET_DIR}/textures/rand_background.obj'))
        return builder.build_static('ground')

    def _setup_viewer(self):
        super()._setup_viewer()
        self._viewer.set_camera_xyz(x=2, y=0, z=1.5)
        self._viewer.set_camera_rpy(r=0, p=-0.3, y=np.pi)

    def evaluate(self, **kwargs) -> dict:
        # Not Implemented
        return {'success': True}

    def compute_normalized_dense_reward(self, **kwargs):
        # Not Implemented
        return 0

    def compute_dense_reward(self, **kwargs):
        # Not Implemented
        return 0

    def get_point_cloud(self, exclude_objs=[]):
        def is_state_unchanged():
            exclude_obj_ids = [obj.model.id for obj in exclude_objs]
            obj_ids = []
            for obj in (self.objs + self.articulations):
                if isinstance(obj.model, sapien.Articulation):
                    obj_ids.append(obj.model.get_links()[0].id)
                else:
                    obj_ids.append(obj.model.id)
            cur_state = self.scene.pack()
            for k1, v1 in cur_state.items():
                for k2, v2 in v1.items():
                    if k2 not in obj_ids or k2 in exclude_obj_ids:
                        continue

                    if k2 not in self.last_env_state[k1] or len(self.last_env_state[k1][k2]) != len(v2):
                        return False
                    for x1, x2 in zip(v2, self.last_env_state[k1][k2]):
                        if abs(x1 - x2) > 1e-3:
                            return False
            return True

        if self.last_env_state is None or not is_state_unchanged():
            obj_meshes = []
            for obj in self.objs + self.articulations:
                if obj not in exclude_objs:
                    obj_meshes.append(obj.get_obj_mesh())
            obj_meshes = merge_meshes(obj_meshes)
            self.pcd_cache = get_point_cloud_from_meshes([obj_meshes])
            self.last_env_state = self.scene.pack()
        else:
            logging.debug('Use cached point cloud')
        return self.pcd_cache.copy()

    def sample_object(self):
        return sample(self.objs, 1)[0]

    @property
    def scene(self):
        return self._scene

    @property
    def sim_steps_per_control(self):
        return self._sim_steps_per_control

    def get_images(self) -> Dict[str, Dict[str, np.ndarray]]:
        imgs = super().get_images()
        for cam, data in imgs.items():
            for tp, img in data.items():
                if tp == 'Color':
                    imgs[cam][tp] = np.clip(img[..., :3] * 255, 0, 255).astype(np.uint8)
                elif tp == 'Position':
                    img[img[..., 3] >= 1, 2] = 0
                    imgs[cam][tp] = (-img[..., 2] * 1000).astype(np.uint16)
                elif tp == 'Segmentation':
                    imgs[cam][tp] = img[..., :2].astype(np.uint8)
        return imgs

    def render(self):
        if self.render_mode is None:
            raise RuntimeError("render_mode is not set.")
        if self.render_mode == "dummy":
            self.update_render()
            return None
        else:
            return super().render()

    def get_objs_by_name(self, name: str):
        return get_entity_by_name(self.objs, name, False)

    def get_articulation_by_name(self, name: str):
        return get_entity_by_name(self.articulations, name, False)

    def get_support_articulation(self, obj):
        contacts = self.scene.get_contacts()
        art_links = []
        art_name = []
        for art in self.articulations:
            if isinstance(art.model, sapien.Articulation):
                art_links.extend(art.model.get_links())
                art_name += [art.name] * len(art.model.get_links())
            else:
                art_links.append(art.model)
                art_name.append(art.name)

        for cont in contacts:
            if cont.actor0 == obj.model:
                another = cont.actor1
            elif cont.actor1 == obj.model:
                another = cont.actor0
            else:
                continue
            if another in art_links:
                idx = art_links.index(another)
                return self.get_articulation_by_name(art_name[idx])
