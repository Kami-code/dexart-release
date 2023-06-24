import torch
import torch.nn as nn

from icecream import ic, install
install()
ic.configureOutput(includeContext=True, contextAbsPath=True, prefix='File ')

class PointNet(nn.Module): # actually pointnet
    def __init__(self, point_channel=3, classes=4):
        super(PointNet, self).__init__()

        print(f'PointNet')

        in_channel = point_channel
        mlp_out_dim = 256
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, mlp_out_dim),
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(mlp_out_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, classes),
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
        N = x.shape[1]
        # Local
        x = self.local_mlp(x)
        local_feats = x
        # gloabal max pooling
        x = torch.max(x, dim=1)[0]
        x = x.view(-1, 1, x.shape[-1]).repeat(1, N, 1)
        # concat local feats
        x = torch.cat([local_feats, x], dim=-1)
        # Output
        x = self.output_mlp(x)
        # Softmax
        x = torch.softmax(x, dim=-1)
        return x


class PointNetMedium(nn.Module): # actually pointnet
    def __init__(self, point_channel=3, classes=4):
        super(PointNetMedium, self).__init__()

        print(f'PointNet')

        in_channel = point_channel
        mlp_out_dim = 256
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, mlp_out_dim),
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(mlp_out_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, classes),
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
        N = x.shape[1]
        # Local
        x = self.local_mlp(x)
        local_feats = x
        # gloabal max pooling
        x = torch.max(x, dim=1)[0]
        x = x.view(-1, 1, x.shape[-1]).repeat(1, N, 1)
        # concat local feats
        x = torch.cat([local_feats, x], dim=-1)
        # Output
        x = self.output_mlp(x)
        # Softmax
        x = torch.softmax(x, dim=-1)
        return x



class PointNetLarge(nn.Module): # actually pointnet
    def __init__(self, point_channel=3, classes=4):
        super(PointNetLarge, self).__init__()

        print(f'PointNet')

        in_channel = point_channel
        mlp_out_dim = 256
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, mlp_out_dim),
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(mlp_out_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, classes),
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
        N = x.shape[1]
        # Local
        x = self.local_mlp(x)
        local_feats = x
        # gloabal max pooling
        x = torch.max(x, dim=1)[0]
        x = x.view(-1, 1, x.shape[-1]).repeat(1, N, 1)
        # concat local feats
        x = torch.cat([local_feats, x], dim=-1)
        # Output
        x = self.output_mlp(x)
        # Softmax
        x = torch.softmax(x, dim=-1)
        return x