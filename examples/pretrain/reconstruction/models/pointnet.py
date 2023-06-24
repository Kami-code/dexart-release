import random

import torch
import torch.nn as nn

import sys, itertools, numpy as np
from chamferdist import ChamferDistance

from icecream import ic, install
install()
ic.configureOutput(includeContext=True, contextAbsPath=True, prefix='File ')


class PointNet(nn.Module): # actually pointnet
    def __init__(self, point_channel=3):
        # NOTE: we require the output dim to be 256, in order to match the pretrained weights
        super(PointNet, self).__init__()

        print(f'PointNet')

        in_channel = point_channel
        mlp_out_dim = 256
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, mlp_out_dim),
        )

        self.reset_parameters_()


    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        '''
        x: [B, N, 3]
        '''
        # N = x.shape[1]
        # Local
        x = self.local_mlp(x)
        # local_feats = x
        # gloabal max pooling
        x = torch.max(x, dim=1)[0]

        return x




class EncoderDecoder(nn.Module):
    def __init__(self, **kwargs):
        super(EncoderDecoder, self).__init__()

        self.grid_size = 4
        self.grid_scale = 0.05
        self.num_coarse = 84
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__dict__.update(kwargs)  # to update args, num_coarse, grid_size, grid_scale

        self.num_fine = self.grid_size ** 2 * self.num_coarse  # 16384
        self.meshgrid = [[-self.grid_scale, self.grid_scale, self.grid_size],
                         [-self.grid_scale, self.grid_scale, self.grid_size]]

        self.pointnet = PointNet()

        # batch normalisation will destroy limit the expression
        self.folding1 = nn.Sequential(
            nn.Linear(256, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_coarse * 3))

        self.folding2 = nn.Sequential(
            nn.Conv1d(256+2+3, 512, 1),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1))

    def build_grid(self, batch_size):
        # a simpler alternative would be: torch.meshgrid()
        x, y = np.linspace(*self.meshgrid[0]), np.linspace(*self.meshgrid[1])
        points = np.array(list(itertools.product(x, y)))
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)

        return torch.tensor(points).float().to(self.device)

    def tile(self, tensor, multiples):
        # substitute for tf.tile:
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/tile
        # Ref: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        def tile_single_axis(a, dim, n_tile):
            init_dim = a.size()[dim]
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*repeat_idx)
            order_index = torch.Tensor(
                np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).long()
            return torch.index_select(a, dim, order_index.to(self.device))

        for dim, n_tile in enumerate(multiples):
            if n_tile == 1:  # increase the speed effectively
                continue
            tensor = tile_single_axis(tensor, dim, n_tile)
        return tensor

    @staticmethod
    def expand_dims(tensor, dim):
        # substitute for tf.expand_dims:
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/expand_dims
        # another solution is: torch.unsqueeze(tensor, dim=dim)
        return tensor.unsqueeze(-1).transpose(-1, dim)

    def forward(self, x):
        # use the same variable naming as:
        # https://github.com/wentaoyuan/pcn/blob/master/models/pcn_cd.py
        feature = self.pointnet(x)

        coarse = self.folding1(feature)
        coarse = coarse.view(-1, self.num_coarse, 3)

        grid = self.build_grid(x.shape[0])
        grid_feat = grid.repeat(1, self.num_coarse, 1)

        point_feat = self.tile(self.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        point_feat = point_feat.view([-1, self.num_fine, 3])

        global_feat = self.tile(self.expand_dims(feature, 1), [1, self.num_fine, 1])
        feat = torch.cat([grid_feat, point_feat, global_feat], dim=2)

        center = self.tile(self.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        center = center.view([-1, self.num_fine, 3])

        fine = self.folding2(feat.transpose(2, 1)).transpose(2, 1) + center

        return coarse, fine


class ChamLoss(nn.Module):
    def __init__(self):
        super(ChamLoss, self).__init__()

    @staticmethod
    def dist_cd(pc2, pc1):
        chamfer_dist = ChamferDistance()
        dist1 = chamfer_dist(pc2, pc1)
        dist2 = chamfer_dist(pc1, pc2)
        return torch.mean(torch.sqrt(dist2)) + torch.mean(torch.sqrt(dist1)) * 0.1

    def forward(self, coarse, fine, gt, alpha):
        return self.dist_cd(coarse, gt) + alpha * self.dist_cd(fine, gt)


if __name__ == '__main__':

    model = EncoderDecoder().to('cuda')
    print(model)
    input_pc = torch.rand(7, 672, 3).to('cuda')
    x = model(input_pc)
    print(x[1].shape)
    # test the chamfer loss
    loss_fn = ChamLoss()
    # transpose the input pc
    # input_pc = input_pc.transpose(1, 2)
    ic(x[0].shape, x[1].shape, input_pc.shape)
    loss = loss_fn(x[0], x[1], input_pc, 0.5)
    print(loss)
    loss.backward()

