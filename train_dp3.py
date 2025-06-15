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
    def __init__(self, data_dir, horizon= 16, n_obs_steps = 2):
        self.samples = []
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps

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

    @staticmethod
    def visualize_pc(demo1, demo2):
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        obs_pc = o3d.geometry.PointCloud()
        obs_pc.points = o3d.utility.Vector3dVector(demo2[3])
        obs_pc.paint_uniform_color([0, 0.6, 1])
        vis.add_geometry(obs_pc)

        imagine_pc = o3d.geometry.PointCloud()
        imagine_pc.points = o3d.utility.Vector3dVector(demo1[3])
        imagine_pc.paint_uniform_color([1.0, 0.6, 0])
        vis.add_geometry(imagine_pc)

        all_points = np.vstack([np.asarray(obs_pc.points), np.asarray(imagine_pc.points)])
        center = all_points.mean(axis=0)

        front = np.array([0.0, 1.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(front, up)

        offset_distance = 0.15  
        shifted_lookat = center + offset_distance * right

        view_ctl = vis.get_view_control()
        view_ctl.set_lookat(shifted_lookat.tolist())
        view_ctl.set_front(front.tolist())
        view_ctl.set_up(up.tolist())
        view_ctl.set_zoom(0.8)

        vis.poll_events()
        vis.update_renderer()
        
        first_frame_path = os.path.join("/home/xinyu/dexart-release/visualization", "stage_3.png")
        vis.capture_screen_image(first_frame_path)

    def getGoal(self, traj, start_idx):
        all_stages = np.array([i['obs']['stage'] for i in traj])
        curr_stage = all_stages[start_idx]

        # Find first occurrence of next stage
        try:
            next_stage_idx = np.where(all_stages == (curr_stage + 1))[0][0]
        except:
            # Might fail to find next stage idx if:
            # - Already at last stage
            # - Skipped one intermediate state
            # In that case, just take the last idx
            next_stage_idx = -1

        goal_obs_imagin = traj[next_stage_idx]["obs"]["imagined_robot_point_cloud"]
        goal_obs_env = traj[next_stage_idx]["obs"]["observed_point_cloud"]

        return goal_obs_imagin, goal_obs_env


    def __getitem__(self, idx):
        traj, start_idx = self.samples[idx]
        obs_window = traj[start_idx - self.n_obs_steps : start_idx]
        action_window = traj[start_idx : start_idx + self.horizon]

        goal_obs_imagin, goal_obs_env = self.getGoal(traj, start_idx)

        obs = {
            'point_cloud': torch.stack([torch.tensor(o["obs"]["observed_point_cloud"], dtype=torch.float32) for o in obs_window]),
            'imagin_robot': torch.stack([torch.tensor(o["obs"]['imagined_robot_point_cloud'], dtype=torch.float32) for o in obs_window]),
            'goal_gripper_pcd': torch.stack([torch.tensor(goal_obs_imagin, dtype=torch.float32)] * self.n_obs_steps),
            'robot0_eef_pos': torch.stack([torch.tensor(o["obs"]['palm_pose.p'], dtype=torch.float32) for o in obs_window]),
            'robot0_eef_quat': torch.stack([torch.tensor(o["obs"]['palm_pose.q'], dtype=torch.float32) for o in obs_window]),
            'robot0_gripper_qpos': torch.stack([torch.tensor(o["obs"]['robot_qpos_vec'][-16:], dtype=torch.float32) for o in obs_window]),
        }

        action = torch.stack([torch.tensor(o["action"], dtype=torch.float32) for o in action_window])
        
        #self.visualize_pc(last_obs_per_stage, last_obs_env)
        #input("Paused. Press Enter to continue...")

        return {
            'obs': obs,
            'action': action
        }


def build_normalizer(dataset):
    obs_accum = collections.defaultdict(list)
    action_accum = []

    for sample in dataset:
        obs = sample["obs"]
        action = sample["action"]

        # Exclude point cloud fields from normalization? CONFIRM
        obs_clean = {k: v for k, v in obs.items() if k not in ['point_cloud', 'imagin_robot', 'goal_gripper_pcd']}

        for k, v in obs_clean.items():
            # v is (n_obs_steps, dim); flatten across time
            obs_accum[k].append(v.reshape(-1, v.shape[-1]))

        # action is (n_action_steps, dim); flatten across time
        action_accum.append(action.reshape(-1, action.shape[-1]))

    obs_stacked = {k: torch.cat(v_list, dim=0) for k, v_list in obs_accum.items()}
    actions_stacked = torch.cat(action_accum, dim=0)

    normalizer = LinearNormalizer()
    normalizer.fit(obs_stacked)

    action_normalizer = LinearNormalizer()
    action_normalizer.fit({"action": actions_stacked})
    normalizer["action"] = action_normalizer["action"]

    return normalizer



@hydra.main(version_base="1.1", config_path="tax3d-conditioned-mimicgen/equi_diffpo/config", config_name="dp3")
def main(cfg):

    data_dir = "/data/xinyu/demo_dexart_Jun13/laptop"
    batch_size = 128
    num_epochs = 50

    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DP3DexArtDataset(data_dir)
    
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)


    shape_meta = {
        'obs': {
            'point_cloud': {'shape': (512, 3)},
            'imagin_robot': {'shape': (96, 3)},
            'goal_gripper_pcd': {'shape': (96, 3)},
            'robot0_eef_pos': {'shape': (3,)},
            'robot0_eef_quat': {'shape': (4,)},        
            'robot0_gripper_qpos': {'shape': (16,)}
        },
        'action': {'shape': (22,)}
    }


    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    horizon = 16
    n_action_steps = 8
    n_obs_steps = 2

    pointcloud_encoder_cfg = cfg.policy.get("pointcloud_encoder_cfg", None)

    model = DP3(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        horizon=horizon,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        pointcloud_encoder_cfg=pointcloud_encoder_cfg,
        pointnet_type="act3d",
        goal_mode='pointcloud_oracle',
    ).to(device)


    normalizer = build_normalizer(dataset)
    model.set_normalizer(normalizer)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    avg_train_losses = []
    avg_val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_val_loss = 0.0
        train_count = 0
        val_count = 0

        for batch in train_loader:
            
            #print("batch shape:")
            #print(batch['obs']['point_cloud'].shape)
            #print(batch['action'].shape)

            obs_batch = batch["obs"]
            action_batch = batch["action"].to(device)

            obs_batch = {k: v.to(device) for k, v in batch["obs"].items()}
            action_batch = batch["action"].to(device)
            #print("obs batch:")
            #print(obs_batch)

            model_input = {"obs": obs_batch, "action": action_batch}
            loss, loss_dict, _ = model.compute_loss(model_input)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_count += 1


        avg_train_loss = total_train_loss / train_count
        avg_train_losses.append(avg_train_loss)

        # ===== Validation =====
        model.eval()
        total_val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                obs_batch = {k: v.to(device) for k, v in batch["obs"].items()}
                action_batch = batch["action"].to(device)

                model_input = {"obs": obs_batch, "action": action_batch}
                loss, _, _ = model.compute_loss(model_input)

                total_val_loss += loss.item()
                val_count += 1

        avg_val_loss = total_val_loss / val_count
        avg_val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        torch.save(model.state_dict(), f"dp3_epoch_{epoch+1}.pt")


    # ===== Final Test Evaluation =====
    model.eval()
    total_test_loss = 0.0
    test_count = 0

    with torch.no_grad():
        for batch in test_loader:
            obs_batch = {k: v.to(device) for k, v in batch["obs"].items()}
            action_batch = batch["action"].to(device)

            model_input = {"obs": obs_batch, "action": action_batch}
            loss, _, _ = model.compute_loss(model_input)

            total_test_loss += loss.item()
            test_count += 1

    avg_test_loss = total_test_loss / test_count
    print(f"Final Test Loss: {avg_test_loss:.4f}")


    '''
    if avg_losses:
        try:
            plt.figure()
            plt.plot(range(1, len(avg_losses) + 1), avg_losses, marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Average Training Loss')
            plt.title('Training Loss vs. Epochs')
            plt.grid(True)
            plt.show()
            #plt.savefig("training_loss_plot.png")
            #plt.close()
            print("Plot saved successfully.")
        except Exception as e:
            print("Failed to plot:", e)
    else:
        print("avg_losses is empty — skipping plot.")
    '''

if __name__ == "__main__":
    main()
