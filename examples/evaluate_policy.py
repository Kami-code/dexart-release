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
    np.random.seed(args.seed)


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
                     pc_noise=True,
                     pc_seg=True,
                     index=indeces,
                     img_type='robot',
                     rand_pos=rand_pos,
                     rand_degree=rand_degree)
    env.seed(args.seed)

    policy = PPO.load(checkpoint_path, env, 'cuda:0',
                      policy_kwargs=get_3d_policy_kwargs(extractor_name='smallpn'),
                      check_obs_space=False, force_load=True)
    policy.set_random_seed(args.seed)

    eval_instances = len(env.instance_list)
    eval_per_instance = args.eval_per_instance
    success_list = list()
    reward_list = list()

    with tqdm(total=eval_per_instance * eval_instances) as pbar:
        for _ in range(eval_per_instance):
            for _ in range(eval_instances):
                obs = env.reset()
                eval_success = False
                reward_sum = 0
                for j in range(env.horizon):
                    if isinstance(obs, dict):
                        for key, value in obs.items():
                            obs[key] = value[np.newaxis, :]
                    else:
                        obs = obs[np.newaxis, :]
                    action = policy.predict(observation=obs, deterministic=True)[0]
                    obs, reward, done, _ = env.step(action)
                    reward_sum += reward
                    if env.is_eval_done:
                        eval_success = True
                    if done:
                        break
                reward_list.append(reward_sum)
                success_list.append(int(eval_success))
                pbar.update(1)
    print(f"checkpoint in {checkpoint_path} success rate = {np.mean(success_list)}")
