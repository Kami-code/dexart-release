


import open3d as o3d
import numpy as np
import pickle
import time
import os

with open("data/outputs/2025.06.13/16.48.17_train_dp3_stack_d1/demo_dp3/laptop/success_demo/demo_2.pkl", "rb") as f:
    demo_data = pickle.load(f)

output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)


vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Demo Playback', width=1920, height=1080)
coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
vis.add_geometry(coord)

# Initialize with first frame
obs_pc = o3d.geometry.PointCloud()
obs_pc.points = o3d.utility.Vector3dVector(demo_data[0]['obs']['observed_point_cloud'])
obs_pc.paint_uniform_color([0, 0.6, 1])
vis.add_geometry(obs_pc)

imagine_pc = o3d.geometry.PointCloud()
imagine_pc.points = o3d.utility.Vector3dVector(demo_data[0]['obs']['imagined_robot_point_cloud'])
imagine_pc.paint_uniform_color([1.0, 0.6, 0])
vis.add_geometry(imagine_pc)


def get_combined_center(pc1, pc2):
    all_points = np.vstack([np.asarray(pc1.points), np.asarray(pc2.points)])
    return all_points.mean(axis=0)

center = get_combined_center(obs_pc, imagine_pc)

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


first_frame_path = os.path.join(output_dir, "frame_0000.png")
vis.capture_screen_image(first_frame_path)


for i, observed in enumerate(demo_data[1:], start=1):
    obs = observed['obs']
    obs_pc.points = o3d.utility.Vector3dVector(obs['observed_point_cloud'])
    imagine_pc.points = o3d.utility.Vector3dVector(obs['imagined_robot_point_cloud'])
    
    vis.update_geometry(obs_pc)
    vis.update_geometry(imagine_pc)
    
    vis.poll_events()
    vis.update_renderer()

    image_path = os.path.join(output_dir, f"frame_{i:04d}.png")
    vis.capture_screen_image(image_path)

    time.sleep(0.13)  # playback speed

print("Playback finished. Close window to exit.")
vis.run()
vis.destroy_window()

