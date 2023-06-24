# train semantic segmentation with GPNet

from models.pointnet import PointNet, SimSiam, D

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid
import os
import sys
import time
from tqdm import tqdm
from icecream import ic, install
install()
ic.configureOutput(includeContext=True, contextAbsPath=True, prefix='File ')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

class Solver(object):
    def __init__(self, config, train_loader, val_loader, test_loader):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = SimSiam(rotate=True).to(self.device)
        # self.load('checkpoints/1.pth')
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'], betas=(0.9, 0.999),
                                            eps=1e-08,
                                            weight_decay=0)
        self.criterion = D

        self.writer = SummaryWriter(config['log_dir'])

    def train(self):
        for epoch in range(self.config['num_epochs']):
            tic = time.time()
            self.model.train()
            for i, (points, _) in tqdm(enumerate(self.train_loader)):  # we do not need labels
                points = points.float().to(self.device)
                self.optimizer.zero_grad()
                z1, z2, p1, p2 = self.model(points)
                loss = self.criterion(p1, z2) / 2 + self.criterion(p2, z1) / 2
                loss.backward()
                self.optimizer.step()
                if i % self.config['log_step'] == 0:
                    # print(f"Epoch [{epoch + 1}/{self.config['num_epochs']}], Step [{i + 1}/{len(self.train_loader)}], Loss: {loss.item():.4f}")
                    self.writer.add_scalar('training loss', loss.item(), epoch * len(self.train_loader) + i)
                    self.writer.add_scalar('learning rate', self.optimizer.param_groups[0]['lr'], epoch * len(self.train_loader) + i)
            self.writer.add_scalar('epoch time', time.time() - tic, epoch)
            if epoch % self.config['save_step'] == 0:
                self.save(epoch)


    def save(self, epoch):
        torch.save(self.model.state_dict(), f"{self.config['log_dir']}/simsiam_{epoch}.pth")
        torch.save(self.model.encoder.state_dict(), f"{self.config['log_dir']}/simsiam_pn_{epoch}.pth")

    def load(self, path):
        self.model.load_state_dict(torch.load(path), strict=True)

    def visualize(self, split):
        import numpy as np
        import open3d as o3d
        import matplotlib.pyplot as plt
        from icecream import ic, install
        install()
        self.model.eval()
        with torch.no_grad():
            if split == 'val':
                loader = self.val_loader
            elif split == 'train':
                loader = self.train_loader
            else:
                loader = self.test_loader
            for i, (points, _) in enumerate(loader):
                ic(points)
                inpoints = points.float().to(self.device)
                # labels = labels.long().to(self.device)

                coarse, fine = self.model(inpoints)
                ic(coarse)
                coarse = coarse.float().cpu().numpy()[0]
                fine = fine.float().cpu().numpy()[0]

                coarse_cloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(coarse))
                # coarse cloud is red
                coarse_cloud.paint_uniform_color([1, 0, 0])
                fine_cloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(fine))
                fine_cloud.paint_uniform_color([0, 1, 0])
                gt_cloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points[0].numpy()))
                gt_cloud.paint_uniform_color([0, 0, 1])
                # obs_cloud.colors = o3d.utility.Vector3dVector(colors[:, 0:3])
                # draw the axis
                coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

                o3d.visualization.draw_geometries([coarse_cloud, coordinate, fine_cloud, gt_cloud])
                # exit()


def main():
    from data_utils import SemSegDataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pn_only', action='store_true', help='use pointnet only', default=False)
    parser.add_argument('--cat', type=str, default='faucet', help='category to train')
    parser.add_argument('--run', type=str, default='1', help='run id')
    parser.add_argument('--use_img', action='store_true', help='use image', default=True)
    parser.add_argument('--eval', action='store_true', help='eval', default=False)

    args = parser.parse_args()


    pn_only = args.pn_only
    cat = args.cat
    run = args.run
    use_img = args.use_img
    point_channel = 3
    config = {
        'num_epochs': 30,
        'log_step': 10,
        'val_step': 1,
        'log_dir': f'log/simsiam/{cat}/{run}',
        'pn_only': pn_only,
        'lr': 1e-3,
        'classes': 4,
        'save_step': 10,
        'cat': cat,
    }

    assert use_img
    train_dataset = SemSegDataset(split='train', point_channel=point_channel, use_img=use_img, root_dir=f'data/{cat}')
    val_dataset = SemSegDataset(split='val', point_channel=point_channel, use_img=use_img, root_dir=f'data/{cat}')
    test_dataset = SemSegDataset(split='test', point_channel=point_channel, use_img=use_img, root_dir=f'data/{cat}')
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=12)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=12)

    solver = Solver(config, train_loader, val_loader, test_loader)
    # if args.eval:
    #     print('eval')
    #     solver.load(config['log_dir'] + '/complete_30.pth')
    #     for m in solver.model.named_modules():
    #         # ic(m)
    #         if isinstance(m[1], nn.LayerNorm):
    #             ic(m[1].weight.var(), m)
    #             ic(m[1].weight, m)
    #
    #     solver.visualize('train')
    #     exit()
    solver.train()
    solver.save(config['num_epochs'])



if __name__ == '__main__':
    main()

    # pointnet = PointNet()
    # pointnet.load_state_dict(torch.load('/home/helin/Code/dexsuite/stable-baselines3/vision_completion/log/gp/toilet/5/complete_pn_0.pth'), strict=True)
    # for m in pointnet.parameters():
    #     ic(m)

