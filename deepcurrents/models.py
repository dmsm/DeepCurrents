import numpy as np
import torch
import torch.nn as nn

from . import utils


class SurfaceModel(nn.Module):

    def __init__(self, bdry=None, rff_sigma=2, dim_z=0, n_data=None,
                 dim_extra_z=0):
        super().__init__()

        self.rff = utils.InputMapping(3, 2048, sigma=rff_sigma)
        self.f = nn.Sequential(
            nn.Linear(2048+dim_z+dim_extra_z, 256),
            nn.Softplus(),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, 1)
        )

        if dim_z > 0 and n_data is not None:
            self.embedding = nn.Embedding(n_data, dim_z)
            nn.init.normal_(self.embedding.weight, std=0.1)

        if bdry is not None:
            self.compute_alpha = lambda x: utils.biot_savart_3d(x, bdry[None])
        else:
            self.compute_alpha = utils.biot_savart_3d

    def forward(self, x, bdry=None, z=None, only_f=False):
        x = x.clamp(-1, 1)
        y = self.rff(x)

        if z is not None:
            if len(z.shape) == 1:
                z = z[None].expand(y.shape[0], -1)
            y = torch.cat([y, z], dim=1)
        f = self.f(y)

        if only_f:
            return f

        df = torch.autograd.grad(outputs=f,
                                 inputs=x,
                                 grad_outputs=torch.ones_like(f),
                                 create_graph=True,
                                 only_inputs=True)[0]
        if bdry is None:
            alpha = self.compute_alpha(x[None])
        else:
            if len(bdry.shape) == 3:
                bdry = bdry[None]
            alpha = self.compute_alpha(x.view(bdry.shape[0], -1, 3), bdry)

        current = df + alpha.flatten(0, 1)

        return {
            'f': f,
            'current': current,
            'df': df,
            'alpha': alpha
        }


class BoundaryEncoder(nn.Module):

    def __init__(self, dim_z=256, n_boundaries=1):
        super().__init__()

        self.encoders = nn.ModuleList(nn.Sequential(
            nn.Conv1d(3, dim_z, kernel_size=5, padding=2,
                      padding_mode='circular'),
            nn.ReLU(),
            nn.Conv1d(dim_z, dim_z, kernel_size=3, padding=1,
                      padding_mode='circular'),
            nn.ReLU(),
            nn.Conv1d(dim_z, dim_z, kernel_size=3, padding=1,
                      padding_mode='circular'),
        ) for _ in range(n_boundaries))

    def forward(self, bdries):
        zs = []
        for encode, bdry in zip(self.encoders, bdries):
            zs.append(encode(bdry.permute(0, 2, 1).contiguous()).mean(-1))
        z = torch.cat(zs, dim=-1)
        return z
