# train semantic segmentation with GPNet

from models.pointnet import PointNet, PointNetMedium, PointNetLarge

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
from icecream import ic, install
install()
ic.configureOutput(includeContext=True, contextAbsPath=True, prefix='File ')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

class Solver(object):
    def __init__(self, config, train_loader, val_loader, test_loader):
        require_attn = True
        point_channel = 3
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.require_attn = require_attn

        if config['arch'] == 'pn':
            self.model = PointNet().to(self.device)
            self.require_attn = False
        elif config['arch'] == 'mpn':
            self.model = PointNetMedium().to(self.device)
            self.require_attn = False
        elif config['arch'] == 'lpn':
            self.model = PointNetLarge().to(self.device)
            self.require_attn = False
        else:
            raise ValueError('Unknown architecture: {}'.format(config['arch']))
        # self.load('checkpoints/1.pth')
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        if config['cat'] == 'bucket':
            print('Using Weighted CrossEntropy loss')
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([3., 1., 1., 1.]).to(self.device))
        else:
            print('Using CrossEntropy loss')
            self.criterion = nn.CrossEntropyLoss()

        self.writer = SummaryWriter(config['log_dir'])

    def train(self):
        self.validate(0)
        for epoch in range(self.config['num_epochs']):
            tic = time.time()
            self.model.train()
            for i, (points, labels) in enumerate(self.train_loader):
                points = points.float().to(self.device)
                labels = labels.long().to(self.device)

                self.optimizer.zero_grad()

                if self.require_attn:
                    outputs, encode_attn, decode_attn = self.model(points)
                    # outputs: [B, N, 4], labels: [B, N]
                    outputs = outputs.permute(0, 2, 1)
                    decode_attn = decode_attn.permute(0, 2, 1)
                    loss = self.criterion(outputs, labels) * (epoch > 5) + self.criterion(decode_attn, labels) * (epoch <= 30)
                else:
                    outputs = self.model(points)
                    # outputs: [B, N, 4], labels: [B, N]
                    outputs = outputs.permute(0, 2, 1)
                    loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                if i % self.config['log_step'] == 0:
                    # print(f"Epoch [{epoch + 1}/{self.config['num_epochs']}], Step [{i + 1}/{len(self.train_loader)}], Loss: {loss.item():.4f}")
                    self.writer.add_scalar('training loss', loss.item(), epoch * len(self.train_loader) + i)
                    self.writer.add_scalar('learning rate', self.optimizer.param_groups[0]['lr'], epoch * len(self.train_loader) + i)

            self.writer.add_scalar('epoch time', time.time() - tic, epoch)
            if epoch % self.config['val_step'] == 0:
                self.validate(epoch, 'val')
                self.validate(epoch, 'test')

            if epoch % self.config['save_step'] == 0:
                self.save(epoch)

    def validate(self, epoch, split='val'):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            if split == 'val':
                loader = self.val_loader
            elif split == 'train':
                loader = self.train_loader
            else:
                loader = self.test_loader
            for i, (points, labels) in enumerate(loader):
                points = points.float().to(self.device)
                labels = labels.long().to(self.device)

                if self.require_attn:
                    outputs, encode_attn, decode_attn = self.model(points)
                else:
                    outputs = self.model(points)
                _, predicted = torch.max(outputs.data, -1)

                total += labels.size(0) * labels.size(1)
                correct += (predicted == labels).sum().item()

            print(f'Accuracy {split} {epoch}: {100 * correct / total}%')
            self.writer.add_scalar(f'{split} accuracy', correct / total, epoch)

    def save(self, epoch):
        torch.save(self.model.state_dict(), f"{self.config['log_dir']}/{self.config['arch']}_{epoch}.pth")

    def load(self, path):
        self.model.load_state_dict(torch.load(path), strict=True)

    def visualize(self, split, num=10):
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
            # shuffle
            loader.dataset.data = loader.dataset.data[torch.randperm(len(loader.dataset.data))]
            for i, (points, labels) in enumerate(loader):
                inpoints = points.float().to(self.device)
                # labels = labels.long().to(self.device)

                if self.require_attn:
                    outputs, encode_attn, decode_attn = self.model(inpoints)
                    ic(decode_attn)
                else:
                    outputs = self.model(inpoints)
                _, predicted = torch.max(outputs.data, -1)
                pc = points.float().cpu().numpy()[0]
                labels = labels.long().cpu().numpy()[0]
                pred_labels = predicted.cpu().numpy()[0]
                colors = plt.get_cmap("tab20")(pred_labels / 4).reshape(-1, 4)

                obs_cloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc[..., 0:3]))
                obs_cloud.colors = o3d.utility.Vector3dVector(colors[:, 0:3])
                # draw the axis
                coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

                o3d.visualization.draw_geometries([obs_cloud, coordinate])
                # exit()
                if i >= num:
                    break


def main():
    from data_utils import SemSegDataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='pn', help='model architecture')
    parser.add_argument('--cat', type=str, default='faucet', help='category to train')
    parser.add_argument('--run', type=str, default='0', help='run id')
    parser.add_argument('--use_img', action='store_true', help='use image', default=False)
    parser.add_argument('--eval', type=str, default=None, help='eval model name e.g. pn_100.pth')
    parser.add_argument('--vis', type=str, default=None, help='visualize model name e.g. pn_100.pth')
    parser.add_argument('--half', action='store_true', help='use half data', default=False)
    args = parser.parse_args()


    arch = args.arch
    cat = args.cat
    run = args.run
    use_img = args.use_img
    point_channel = 3
    config = {
        'num_epochs': 100,
        'log_step': 10,
        'val_step': 1,
        'log_dir': f'log/segmentation/{arch}/{cat}/{run}',
        'arch': arch,
        'lr': 1e-3,
        'classes': 4,
        'save_step': 20,
        'cat': cat,
    }

    assert use_img
    train_dataset = SemSegDataset(split='train', point_channel=point_channel, use_img=use_img, root_dir=f'data/{cat}', half=args.half)
    val_dataset = SemSegDataset(split='val', point_channel=point_channel, use_img=use_img, root_dir=f'data/{cat}')
    test_dataset = SemSegDataset(split='test', point_channel=point_channel, use_img=use_img, root_dir=f'data/{cat}')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=12)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=12)

    solver = Solver(config, train_loader, val_loader, test_loader)
    # solver.load(f'log/{arch}/{cat}/{run}/{arch}_100.pth')
    # solver.visualize('test')
    if args.eval is not None:
        solver.load(f'{config["log_dir"]}/{args.eval}')
        pass
    elif args.vis is not None:
        solver.load(f'{config["log_dir"]}/{args.vis}')
        solver.visualize('test')
    else:
        solver.train()
        solver.save(config['num_epochs'])


# def eval():
#     from data_utils import SemSegDataset
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--pn_only', action='store_true', help='use pointnet only', default=False)
#     parser.add_argument('--cat', type=str, default='faucet', help='category to train')
#     parser.add_argument('--run', type=str, default='0', help='run id')
#     parser.add_argument('--use_img', action='store_true', help='use image', default=False)
#
#     args = parser.parse_args()
#
#
#     pn_only = args.pn_only
#     cat = args.cat
#     run = args.run
#     use_img = args.use_img
#     point_channel = 3
#     config = {
#         'num_epochs': 1000,
#         'log_step': 10,
#         'val_step': 1,
#         'log_dir': f'log/{"pn" if pn_only else "gp"}/{cat}/{run}',
#         'pn_only': pn_only,
#         'lr': 1e-3,
#         'classes': 4,
#         'save_step': 5,
#         'cat': cat,
#     }
#
#     assert use_img
#     train_dataset = SemSegDataset(split='train', point_channel=point_channel, use_img=use_img, root_dir=f'data/{cat}_img')
#     val_dataset = SemSegDataset(split='val', point_channel=point_channel, use_img=use_img, root_dir=f'data/{cat}_img')
#     test_dataset = SemSegDataset(split='test', point_channel=point_channel, use_img=use_img, root_dir=f'data/{cat}_img')
#     train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=12)
#     val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=12)
#     test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=12)
#
#     solver = Solver(config, train_loader, val_loader, test_loader)
#
#
#     solver.load('checkpoints' + '/toilet_pn_180.pth')
#     for m in solver.model.named_modules():
#         # ic(m)
#         if isinstance(m[1], nn.LayerNorm):
#             ic(m[1].weight.var(), m)
#             ic(m[1].weight, m)
#
#     solver.validate(0, 'val')
#     solver.visualize('train')


if __name__ == '__main__':
    main()
    # eval()
