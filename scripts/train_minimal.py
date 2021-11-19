import argparse
import datetime
import os

import imageio
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from deepcurrents import models, utils


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    t = torch.linspace(0, 2*np.pi, 101)[0:-1]
    if args.boundary == 'hopf':
        # Hopf link
        bdry_verts = torch.stack([
            torch.stack([0.6 * torch.cos(t) - 0.3,
                         0.6 * torch.sin(t),
                         torch.zeros(100)], dim=-1),
            torch.stack([0.6 * torch.cos(t) + 0.3,
                         torch.zeros(100),
                         0.6 * torch.sin(t)], dim=-1)
        ], dim=0).to(device)
    elif args.boundary == 'trefoil':
        # Trefoil knot
        bdry_verts = (1/6) * torch.stack(
            [torch.sin(t) + 2 * torch.sin(2 * t),
             torch.cos(t) - 2 * torch.cos(2 * t),
             -torch.sin(3 * t)], dim=-1).to(device)[None]
    elif args.boundary == 'borromean':
        # Borromean rings
        bdry_verts = torch.stack([
            torch.stack([torch.zeros(100),
                         0.6 * torch.cos(t),
                         0.3 * torch.sin(t)], dim=-1),
            torch.stack([0.3 * torch.cos(t),
                         torch.zeros(100),
                         0.6 * torch.sin(t)], dim=-1),
            torch.stack([0.6 * torch.cos(t),
                         0.3 * torch.sin(t),
                         torch.zeros(100)], dim=-1)
        ], dim=0).to(device)
    else:
        raise ValueError("unsupported boundary type.")

    model = models.SurfaceModel(bdry=bdry_verts, rff_sigma=args.rff_sigma)
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.6)

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    writer = SummaryWriter(os.path.join(
        args.out, datetime.datetime.now().strftime('train-%m%d%y-%H%M%S')),
        flush_secs=1)

    it = 0
    pbar = tqdm(total=args.n_iterations)
    for it in range(args.n_iterations):
        optimizer.zero_grad()

        x = torch.empty(
            args.n_samples, 3).uniform_(-1, 1).to(device)
        x.requires_grad = True

        out = model(x)
        loss = out['current'].norm(p=2, dim=-1).mean()

        loss.backward()
        optimizer.step()
        if it % 10000 == 0:
            scheduler.step()

        pbar.set_postfix({
            'loss': loss.item(),
        })
        pbar.update(1)
        writer.add_scalar('loss', loss.item(), it)

        if it % 10000 == 0:
            torch.save({
                'model': model.state_dict(),
                'opt_state_dict': optimizer.state_dict(),
                'sched_state_dict': scheduler.state_dict(),
                'it': it,
                'boundary': bdry_verts
            }, os.path.join(args.out, f'{it}.pth'))
        it += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='out/minimal')
    parser.add_argument('--n_samples', type=int, default=2**12)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--n_iterations', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--boundary', type=str, default='trefoil')

    parser.add_argument('--rff_sigma', type=float, default=2)

    args = parser.parse_args()
    main(args)
