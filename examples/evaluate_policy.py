import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from dexart.env.task_setting import TRAIN_CONFIG, RANDOM_CONFIG
import numpy as np
from dexart.env.create_env import create_env
from stable_baselines3 import PPO
from examples.train import get_3d_policy_kwargs
from tqdm import tqdm
import pickle

def get_obs(obs):
    state = obs['state']  # shape (32,)
    dp3_obs = {
        'robot_qpos_vec': state[:22],
        'palm_v': state[22: 25],
        'palm_w': state[25: 28],
        'palm_pose.p': state[28: 31],
        "palm_pose.q": obs['quat_obs'],
        'observed_point_cloud': obs['instance_1-point_cloud'],  # (512, 3)
        "observed_pc_seg-gt": obs['instance_1-seg_gt'],         # (512, 4)
        'imagined_robot_point_cloud': obs['imagination_robot'][:, :3],  # (96, 3)
        'imagined_robot_pc_seg-gt': obs['imagination_robot'][:, 3:],    # (96, 4)
        'stage': obs['stage'],
        'progress': obs['progress']
    }
    return dp3_obs



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--eval_per_instance', type=int, default=10)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--use_test_set', dest='use_test_set', action='store_true', default=False)
    args = parser.parse_args()
    task_name = args.task_name
    use_test_set = args.use_test_set
    checkpoint_path = args.checkpoint_path
    np.random.seed(args.seed)       # fix seeds here
    

    if use_test_set:
        indeces = TRAIN_CONFIG[task_name]['unseen']
        print(f"using unseen instances {indeces}")
    else:
        indeces = TRAIN_CONFIG[task_name]['seen']
        print(f"using seen instances {indeces}")

    rand_pos = RANDOM_CONFIG[task_name]['rand_pos']
    rand_degree = RANDOM_CONFIG[task_name]['rand_degree']
    env = create_env(task_name=task_name,
                     use_visual_obs=True,
                     use_gui=True,
                     is_eval=True,
                     pc_noise=True,         # turn off noise
                     pc_seg=True,
                     index=indeces,
                     img_type='robot',
                     rand_pos=rand_pos,         # rand_pos = 0.0 -> no position randomization
                     rand_degree=rand_degree)   # rand_degree = 0.0 -> no rotation randomization
    env.seed(args.seed)         # fix seeds globally

    policy = PPO.load(checkpoint_path, env, 'cuda:0',
                      policy_kwargs=get_3d_policy_kwargs(extractor_name='smallpn'),
                      check_obs_space=False, force_load=True)
    policy.set_random_seed(args.seed)

    eval_instances = len(env.instance_list)
    eval_per_instance = args.eval_per_instance
    success_list = list()
    reward_list = list()

    demo_save_dir = os.path.join('demo_DexArt_w.seg&q', task_name)
    os.makedirs(demo_save_dir, exist_ok=True)

    demo_id = 0

    with tqdm(total=eval_per_instance * eval_instances) as pbar:
        for _ in range(eval_per_instance):       # Loop over number of episodes per instance
            for _ in range(eval_instances):
                obs = env.reset()
                eval_success = False
                reward_sum = 0
                demo_data = []

                for j in range(env.horizon):         # Loop for max steps
                    if isinstance(obs, dict):
                        for key, value in obs.items():
                            if isinstance(value, np.ndarray):
                                obs[key] = value[np.newaxis, :]
                            else:
                                obs[key] = np.array([[value]]) 
                    else:
                        obs = obs[np.newaxis, :]

                    #Use policy to predict next action (deterministic to reduce variance)
                    action = policy.predict(observation=obs, deterministic=True)[0]
                    
                    # Take action and receive next obs and reward
                    obs, reward, done, _ = env.step(action)
                    reward_sum += reward

                    observed = {
                        'obs': get_obs(obs),
                        'action': action,  # shape (22,)
                        'reward': reward
                    }
                    demo_data.append(observed)

                    # Observation Structure:
                    #     instance_1-seg_gt: shape=(512, 4), dtype=float64
                    #     instance_1-point_cloud: shape=(512, 3), dtype=float64
                    #     imagination_robot: shape=(96, 7), dtype=float64
                    #     state: shape=(32,), dtype=float64
                    #     oracle_state: shape=(32,), dtype=float64

                    
                    # print("Observation Structure:")
                    # for key, value in obs.items():
                    #      print(f"{key}: shape={value.shape}, dtype={value.dtype}")
                    # break
                    
                    # # Extract and concatenate point cloud data from obs
                    # observed_pc = np.concatenate([S
                    #     obs['instance_1-point_cloud'],    # (512, 3) point cloud
                    #     obs['instance_1-seg_gt']          # (512, 4) segmentation label
                    # ], axis=1)                             # (512, 7) 
                    # observed_pc = np.concatenate([
                    #     observed_pc,
                    #     obs['imagination_robot']           # (96, 7)
                    # ], axis=0)                             # => (608, 7)
                    # assert observed_pc.shape == (608, 7)
                    # demo_data.append(observed_pc)  # Append to trajectory list 
                    

                    if env.is_eval_done:
                        eval_success = True
                    
                    if done:
                        break
                
                reward_list.append(reward_sum)
                success_list.append(int(eval_success))
                pbar.update(1)

                if eval_success:
                    #print(demo_data)         
                    with open(os.path.join(demo_save_dir, f'demo_{demo_id}.pkl'), 'wb') as f:
                        pickle.dump(demo_data, f)

                    demo_id += 1


    print(f"checkpoint in {checkpoint_path} success rate = {np.mean(success_list)}")
    print(f"Saved {demo_id} successful demos to: {demo_save_dir}")
