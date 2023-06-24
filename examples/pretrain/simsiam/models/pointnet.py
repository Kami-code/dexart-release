import torch
import torch.nn as nn

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


class SimSiam(nn.Module):
    def __init__(self, rotate=False):
        super(SimSiam, self).__init__()

        self.encoder = PointNet()
        self.predictor = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        self.rotate = rotate
        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def point_augment(self, x):
        '''
        x: [B, N, 3]
        '''
        # (1) point cloud jittering
        x = x + torch.randn_like(x) * 0.01
        # (2) point cloud dropout
        mask = torch.rand(x.shape[0], x.shape[1], 1).to(x.device) > 0.5
        x = x * mask
        # (3) 5 degree random rotation
        if self.rotate:
            # rotate along z axis
            theta = torch.rand(x.shape[0], 1, 1).to(x.device) * 2 * 3.14159
            rot_mat = torch.cat([torch.cos(theta), -torch.sin(theta), torch.zeros_like(theta),
                                torch.sin(theta), torch.cos(theta), torch.zeros_like(theta),
                                torch.zeros_like(theta), torch.zeros_like(theta), torch.ones_like(theta)], dim=2).view(-1, 3, 3)
            x = torch.bmm(x, rot_mat)
        return x



    def forward(self, x):
        '''
        x: [B, N, 3]
        '''
        # augment
        x1 = self.point_augment(x)
        x2 = self.point_augment(x)
        # encode
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        # predict
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        # normalize
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        p1 = nn.functional.normalize(p1, dim=1)
        p2 = nn.functional.normalize(p2, dim=1)

        return z1, z2, p1, p2


def D(p, z):
    # negative cosine similarity loss
    loss = -nn.CosineSimilarity(dim=1)(p, z.detach()).mean()
    return loss



def test():
    model = SimSiam(rotate=True)
    x = torch.randn(2, 1024, 3)
    z1, z2, p1, p2 = model(x)
    ic(z1.shape, z2.shape, p1.shape, p2.shape)

if __name__ == '__main__':
    test()