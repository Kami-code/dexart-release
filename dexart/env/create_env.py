import os

import numpy as np
import sapien.core as sapien

from dexart.env.rl_env.faucet_env import FaucetRLEnv
from dexart.env.rl_env.bucket_env import BucketRLEnv
from dexart.env.rl_env.laptop_env import LaptopRLEnv
from dexart.env.rl_env.toilet_env import ToiletRLEnv
from dexart.env import task_setting
from dexart.env.sim_env.constructor import add_default_scene_light


def create_env(task_name, use_visual_obs, use_gui=False, is_eval=False, pc_seg=False,
               pc_noise=False, index=-1, img_type=None, rand_pos=0.0, rand_degree=0, frame_skip=10,
               **kwargs):
    robot_name = "allegro_hand_xarm6_wrist_mounted_face_front"
    rotation_reward_weight = 1
    env_params = dict(robot_name=robot_name, rotation_reward_weight=rotation_reward_weight,
                      use_visual_obs=use_visual_obs, use_gui=use_gui, no_rgb=True, use_old_api=True,
                      index=index, frame_skip=frame_skip, rand_pos=rand_pos, rand_orn=rand_degree / 180 * np.pi,
                      **kwargs)
    if img_type:
        assert img_type in task_setting.IMG_CONFIG.keys()

    if is_eval:
        env_params["no_rgb"] = False
        env_params["need_offscreen_render"] = True

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    if task_name == 'faucet':
        env = FaucetRLEnv(**env_params, friction=5)
    elif task_name == 'bucket':
        env = BucketRLEnv(**env_params, friction=0)
    elif task_name == 'laptop':
        env = LaptopRLEnv(**env_params, friction=5)
    elif task_name == 'toilet':
        env = ToiletRLEnv(**env_params, friction=5)
    else:
        raise NotImplementedError
    if use_visual_obs:
        current_setting = task_setting.CAMERA_CONFIG[task_name]
        env.setup_camera_from_config(current_setting)
        if pc_seg:
            env.setup_visual_obs_config(task_setting.OBS_CONFIG["instance_pc_seg"])
        elif pc_noise:
            env.setup_visual_obs_config(task_setting.OBS_CONFIG["instance_noise"])
        else:
            env.setup_visual_obs_config(task_setting.OBS_CONFIG["instance"])
        if img_type:
            # Specify imagination
            env.setup_imagination_config(task_setting.IMG_CONFIG[img_type])
    if is_eval:
        env.setup_camera_from_config(task_setting.CAMERA_CONFIG["viz_only"])
        add_default_scene_light(env.scene, env.renderer)


    # flush cache
    env.action_space
    env.observation_space
    return env


def create_rgb_env(task_name, use_visual_obs, use_gui=False, is_eval=False, index=-1, rand_pos=0.0, rand_degree=0,
                   frame_skip=10,
                   **kwargs):
    robot_name = "allegro_hand_xarm6_wrist_mounted_face_front"
    rotation_reward_weight = 1
    env_params = dict(robot_name=robot_name, rotation_reward_weight=rotation_reward_weight,
                      use_visual_obs=use_visual_obs, use_gui=use_gui, no_rgb=False,
                      need_offscreen_render=True, rand_orn=rand_degree / 180 * np.pi,
                      index=index, frame_skip=frame_skip, rand_pos=rand_pos, **kwargs)

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    if task_name == 'faucet':
        env = FaucetRLEnv(**env_params, friction=5)
    elif task_name == 'bucket':
        env = BucketRLEnv(**env_params, friction=0)
    elif task_name == 'laptop':
        env = LaptopRLEnv(**env_params, friction=5)
    elif task_name == 'toilet':
        env = ToiletRLEnv(**env_params, friction=5)
    else:
        raise NotImplementedError

    add_default_scene_light(env.scene, env.renderer)
    if use_visual_obs:
        current_setting = task_setting.CAMERA_CONFIG[task_name]
        current_setting['instance_1']['resolution'] = (64, 64)
        env.setup_camera_from_config(current_setting)
        env.setup_visual_obs_config(task_setting.OBS_CONFIG["instance_rgb"])
    if is_eval:
        env.setup_camera_from_config(task_setting.CAMERA_CONFIG["viz_only"])

    # flush cache
    env.action_space
    env.observation_space
    return env


def create_robustness_rgb_env(task_name, use_visual_obs, phi_delta, theta_delta, use_gui=False, is_eval=False,
                              index=-1, rand_pos=0.0, rand_degree=0, frame_skip=10, **kwargs):
    robot_name = "allegro_hand_xarm6_wrist_mounted_face_front"
    rotation_reward_weight = 1
    env_params = dict(robot_name=robot_name, rotation_reward_weight=rotation_reward_weight,
                      use_visual_obs=use_visual_obs, use_gui=use_gui, no_rgb=False,
                      need_offscreen_render=True, rand_orn=rand_degree / 180 * np.pi,
                      index=index, frame_skip=frame_skip, rand_pos=rand_pos, **kwargs)

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    if task_name == 'faucet':
        env = FaucetRLEnv(**env_params, friction=5)
    elif task_name == 'bucket':
        env = BucketRLEnv(**env_params, friction=0)
    elif task_name == 'laptop':
        env = LaptopRLEnv(**env_params, friction=5)
    elif task_name == 'toilet':
        env = ToiletRLEnv(**env_params, friction=5)
    else:
        raise NotImplementedError

    add_default_scene_light(env.scene, env.renderer)

    if use_visual_obs:
        current_setting = task_setting.CAMERA_CONFIG[task_name]
        current_setting['instance_1']['resolution'] = (64, 64)

        robustness_init_camera_config = task_setting.ROBUSTNESS_INIT_CAMERA_CONFIG[task_name]
        r = robustness_init_camera_config['r']
        phi = robustness_init_camera_config['phi'] + phi_delta
        theta = robustness_init_camera_config['theta'] + theta_delta
        center = robustness_init_camera_config['center']

        x0, y0, z0 = center
        # phi in [0, pi/2]
        # theta in [0, 2 * pi]
        x = x0 + r * np.sin(phi) * np.cos(theta)
        y = y0 + r * np.sin(phi) * np.sin(theta)
        z = z0 + r * np.cos(phi)

        cam_pos = np.array([x, y, z])
        forward = np.array([x0 - x, y0 - y, z0 - z])
        forward /= np.linalg.norm(forward)

        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)

        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos

        final_pose = sapien.Pose.from_transformation_matrix(mat44)

        current_setting['instance_1']['pose'] = final_pose
        env.setup_camera_from_config(current_setting, use_opencv_trans=False)
        env.setup_visual_obs_config(task_setting.OBS_CONFIG["instance_rgb"])

        vis_dict = {"instance_viz": current_setting['instance_1']}
        vis_dict['instance_viz']['resolution'] = (1000, 1000)
        env.setup_camera_from_config(vis_dict, use_opencv_trans=False)

    if is_eval:
        env.setup_camera_from_config(task_setting.CAMERA_CONFIG["viz_only"])

    # flush cache
    env.action_space
    env.observation_space
    return env


def create_robustness_env(task_name, use_visual_obs, use_gui=False, is_eval=False, pc_seg=False,
                          pc_noise=False, index=-1, img_type=None, rand_pos=0.0, rand_degree=0.0,
                          phi_delta=0, theta_delta=0, frame_skip=10,
                          **kwargs):
    robot_name = "allegro_hand_xarm6_wrist_mounted_face_front"
    rotation_reward_weight = 1
    env_params = dict(robot_name=robot_name, rotation_reward_weight=rotation_reward_weight,
                      use_visual_obs=use_visual_obs, use_gui=use_gui, no_rgb=True, use_old_api=True,
                      index=index, frame_skip=frame_skip, rand_pos=rand_pos, rand_orn=rand_degree / 180 * np.pi,
                      **kwargs)
    if img_type:
        assert img_type in task_setting.IMG_CONFIG.keys()

    if is_eval:
        env_params["no_rgb"] = False
        env_params["need_offscreen_render"] = True

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    if task_name == 'faucet':
        env = FaucetRLEnv(**env_params, friction=5)
    elif task_name == 'bucket':
        env = BucketRLEnv(**env_params, friction=0)
    elif task_name == 'laptop':
        env = LaptopRLEnv(**env_params, friction=5)
    elif task_name == 'toilet':
        env = ToiletRLEnv(**env_params, friction=5)
    else:
        raise NotImplementedError

    add_default_scene_light(env.scene, env.renderer)
    if use_visual_obs:
        current_setting = task_setting.CAMERA_CONFIG[task_name]

        robustness_init_camera_config = task_setting.ROBUSTNESS_INIT_CAMERA_CONFIG[task_name]
        r = robustness_init_camera_config['r']
        phi = robustness_init_camera_config['phi'] + phi_delta
        theta = robustness_init_camera_config['theta'] + theta_delta
        center = robustness_init_camera_config['center']

        x0, y0, z0 = center
        # phi in [0, pi/2]
        # theta in [0, 2 * pi]
        x = x0 + r * np.sin(phi) * np.cos(theta)
        y = y0 + r * np.sin(phi) * np.sin(theta)
        z = z0 + r * np.cos(phi)

        cam_pos = np.array([x, y, z])
        forward = np.array([x0 - x, y0 - y, z0 - z])
        forward /= np.linalg.norm(forward)

        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)

        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos

        final_pose = sapien.Pose.from_transformation_matrix(mat44)

        current_setting['instance_1']['pose'] = final_pose
        env.setup_camera_from_config(current_setting, use_opencv_trans=False)

        vis_dict = {"instance_viz": current_setting['instance_1']}
        vis_dict['instance_viz']['resolution'] = (1000, 1000)
        env.setup_camera_from_config(vis_dict, use_opencv_trans=False)
        if pc_seg:
            env.setup_visual_obs_config(task_setting.OBS_CONFIG["instance_pc_seg"])
        elif pc_noise:
            env.setup_visual_obs_config(task_setting.OBS_CONFIG["instance_noise"])
        else:
            env.setup_visual_obs_config(task_setting.OBS_CONFIG["instance"])
        if img_type:
            # Specify imagination
            env.setup_imagination_config(task_setting.IMG_CONFIG[img_type])
    if is_eval:
        env.setup_camera_from_config(task_setting.CAMERA_CONFIG["viz_only"])

    # flush cache
    env.action_space
    env.observation_space
    return env
