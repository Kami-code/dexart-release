import numpy as np
import os.path
from glob import glob
from dexart.env.task_setting import TRAIN_CONFIG, RANDOM_CONFIG
from dexart.env.create_env import create_env
from tqdm import tqdm
import argparse


def gen_single_data(task_name, index, split, n_fold=32, img_type='robot', save_path='data/'):
    env = create_env(task_name=task_name,
                     use_visual_obs=True,
                     use_gui=False,
                     is_eval=False,
                     pc_noise=True,
                     pc_seg=True,
                     index=[index],
                     img_type=img_type,
                     rand_pos=RANDOM_CONFIG[task_name]['rand_pos'],
                     rand_degree=RANDOM_CONFIG[task_name]['rand_degree'],
                     )
    obs = env.reset()
    pc_data = []
    for i in tqdm(range(env.horizon * n_fold)):
        action = np.random.uniform(-2, 2, size=env.action_space.shape)
        obs, reward, done, _ = env.step(action)

        qlimits = env.instance.get_qlimits()
        # random qpos
        qpos = np.random.uniform(qlimits[:, 0], qlimits[:, 1])
        env.instance.set_qpos(qpos)

        observed_pc = np.concatenate([obs['instance_1-point_cloud'], obs['instance_1-seg_gt']], axis=1)
        observed_pc = np.concatenate([observed_pc, obs['imagination_robot']], axis=0)
        assert observed_pc.shape == (608, 7)
        pc_data.append(observed_pc)
        env.scene.update_render()

        if i % env.horizon == 0:
            obs = env.reset()  # reset random position and orn each fold

    # save data
    if not os.path.exists(os.path.join(save_path, f"{task_name}")):
        os.makedirs(os.path.join(save_path, f"{task_name}"))
    np.save(os.path.join(save_path, f"{task_name}/{split}_{index}.npy"), pc_data)
    print(f"save {split}_{index}.npy")


def merge_data(category, save_path='data', merge_half=False):
    dir_name = os.path.join(save_path, category)

    dataset = []
    if merge_half:
        for index in TRAIN_CONFIG[category + '_half']['seen']:
            f = os.path.join(dir_name, f"train_{index}.npy")
            data = np.load(f)
            dataset.append(data)
            print(f)
        # save dataset
        dataset = np.concatenate(dataset, axis=0)
        np.save(os.path.join(dir_name, 'train_half.npy'), dataset)

    dataset = []
    for f in glob(f'{dir_name}/train_*.npy'):
        data = np.load(f)
        dataset.append(data)
        print(f)
    # save dataset
    dataset = np.concatenate(dataset, axis=0)
    np.save(os.path.join(dir_name, 'train.npy'), dataset)

    dataset = []
    for f in glob(f'{dir_name}/val_*.npy'):
        data = np.load(f)
        dataset.append(data)
        print(f)
    # save dataset
    dataset = np.concatenate(dataset, axis=0)
    np.save(os.path.join(dir_name, 'val.npy'), dataset)

    dataset = []
    for f in glob(f'{dir_name}/test_*.npy'):
        data = np.load(f)
        dataset.append(data)
        print(f)
    # save dataset
    dataset = np.concatenate(dataset, axis=0)
    np.save(os.path.join(dir_name, 'test.npy'), dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='faucet')
    args = parser.parse_args()
    task_name = args.task_name

    for index in TRAIN_CONFIG[task_name]['seen']:
        gen_single_data(task_name, index, 'train', n_fold=32)
    for index in TRAIN_CONFIG[task_name]['seen']:
        gen_single_data(task_name, index, 'val', n_fold=4)
    for index in TRAIN_CONFIG[task_name]['unseen']:
        gen_single_data(task_name, index, 'test', n_fold=4)
    merge_data(task_name)
