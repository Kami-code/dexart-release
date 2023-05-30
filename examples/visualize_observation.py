from sapien.utils import Viewer
from examples.utils import get_viewpoint_camera_parameter, visualize_observation
import argparse
import numpy as np
from dexart.env.create_env import create_env
import open3d as o3d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, required=True)
    args = parser.parse_args()
    task_name = args.task_name

    env = create_env(task_name=task_name, use_visual_obs=True, img_type='robot', use_gui=True)
    robot_dof = env.robot.dof
    env.seed(0)
    obs = env.reset()

    # config the viewer
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    viewer.focus_camera(env.cameras['instance_1'])
    env.viewer = viewer

    origin, target, up, m44 = get_viewpoint_camera_parameter()

    for i in range(10):
        action = np.random.rand(22)
        obs, reward, done, info = env.step(action)
        viewer.render()


    pc = visualize_observation(obs, use_seg=False, img_type='robot')
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pc, coordinate], zoom=1,
                                      front=origin - target,
                                      lookat=target,
                                      up=up)

    viewer.close()