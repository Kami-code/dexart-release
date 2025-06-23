import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim

import sys
sys.path.append('tax3d-conditioned-mimicgen')
from equi_diffpo.policy.dp3 import DP3
from equi_diffpo.model.common.normalizer import LinearNormalizer
from diffusers.schedulers import DDPMScheduler
import os
import hydra
import collections
import open3d as o3d
import numpy as np

#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt

class DP3DexArtDataset(Dataset):
    def __init__(self, data_dir, horizon= 16, n_obs_steps = 2, goal_mode="None"):
        self.samples = []
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.goal_mode = goal_mode

        for fname in os.listdir(data_dir):
            if fname.endswith(".pkl"):
                with open(os.path.join(data_dir, fname), "rb") as f:
                    traj = pickle.load(f)
                T = len(traj)
                max_start = T - horizon + 1
                for t in range(n_obs_steps, max_start):
                    self.samples.append((traj, t))

    def __len__(self):
        return len(self.samples)

    def getGoal(self, traj, start_idx):
        try:
            progress_array = np.array([i['obs']['progress'] for i in traj])
            subgoal_idx = np.where(progress_array > 1e-5)[0][0]
        except:
            subgoal_idx = -1

        if start_idx < subgoal_idx:
            next_stage_idx = subgoal_idx
        else:
            next_stage_idx = -1

        goal_obs_imagin = traj[next_stage_idx]["obs"]["imagined_robot_point_cloud"]
        goal_obs_env = traj[next_stage_idx]["obs"]["observed_point_cloud"]

        return goal_obs_imagin, goal_obs_env
    

    def __getitem__(self, idx):
        traj, start_idx = self.samples[idx]
        obs_window = traj[start_idx - self.n_obs_steps : start_idx]
        action_window = traj[start_idx : start_idx + self.horizon]

        if self.goal_mode == 'pointcloud_oracle':
            goal_obs_imagin, goal_obs_env = self.getGoal(traj, start_idx)
            obs = {
                'point_cloud': torch.stack([torch.tensor(o["obs"]["observed_point_cloud"], dtype=torch.float32) for o in obs_window]),
                'imagin_robot': torch.stack([torch.tensor(o["obs"]['imagined_robot_point_cloud'], dtype=torch.float32) for o in obs_window]),
                'goal_gripper_pcd': torch.stack([torch.tensor(goal_obs_imagin, dtype=torch.float32)] * self.n_obs_steps),
                'robot0_eef_pos': torch.stack([torch.tensor(o["obs"]['palm_pose.p'], dtype=torch.float32) for o in obs_window]),
                'robot0_eef_quat': torch.stack([torch.tensor(o["obs"]['palm_pose.q'], dtype=torch.float32) for o in obs_window]),
                'robot0_gripper_qpos': torch.stack([torch.tensor(o["obs"]['robot_qpos_vec'][-16:], dtype=torch.float32) for o in obs_window]),
            }
            #print("pointc")
        elif self.goal_mode == 'None':
            obs = {
                'point_cloud': torch.stack([torch.tensor(o["obs"]["observed_point_cloud"], dtype=torch.float32) for o in obs_window]),
                'imagin_robot': torch.stack([torch.tensor(o["obs"]['imagined_robot_point_cloud'], dtype=torch.float32) for o in obs_window]),
                'goal_gripper_pcd': torch.stack([torch.tensor(o["obs"]['imagined_robot_point_cloud'], dtype=torch.float32) for o in obs_window]),
                'robot0_eef_pos': torch.stack([torch.tensor(o["obs"]['palm_pose.p'], dtype=torch.float32) for o in obs_window]),
                'robot0_eef_quat': torch.stack([torch.tensor(o["obs"]['palm_pose.q'], dtype=torch.float32) for o in obs_window]),
                'robot0_gripper_qpos': torch.stack([torch.tensor(o["obs"]['robot_qpos_vec'][-16:], dtype=torch.float32) for o in obs_window]),
            }
            #print("null")

        action = torch.stack([torch.tensor(o["action"], dtype=torch.float32) for o in action_window])

        return {
            'obs': obs,
            'action': action
        }
