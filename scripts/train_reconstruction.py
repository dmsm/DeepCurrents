import argparse
import datetime
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from deepcurrents import models, utils, datasets
from deepcurrents.metric import Metric


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load data
    dataset = datasets.MeshDataset(args.data, n_samples=args.n_samples,
                                   idx=args.idx)
    dataloader = datasets.MultiEpochsDataLoader(dataset, batch_size=1,
                                                num_workers=8)

    data = dataset[0]
    verts = data['verts'].to(device)
    faces = data['faces'].to(device)
    face_normals = data['face_normals'].to(device)
    bdry_verts = data['bdry_verts'].to(device)
    bdry_edges = data['bdry_edges'].to(device)

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

    metric = getattr(Metric, args.metric)

    pbar = tqdm(total=args.n_iterations)
    it = 0
    done = False
    while not done:
        for batch in dataloader:
            optimizer.zero_grad()

            x_inside = batch['x_inside'][0].to(device)
            x_outside = batch['x_outside'][0].to(device)
            x_uniform = torch.empty(
                args.n_samples, 3).uniform_(-1, 1).to(device)
            x = torch.cat([x_uniform, x_inside, x_outside])
            x.requires_grad = True

            with torch.no_grad():
                sq_dists_bdry = utils.point_edge_distance(
                    x_uniform[:args.n_samples//2],
                    torch.tensor([0]).to(device),
                    bdry_edges,
                    torch.tensor([0]).to(device),
                    args.n_samples
                )[0]
                weight = torch.exp(-sq_dists_bdry / (2 * 0.1**2))
                weight = torch.cat([weight, torch.ones_like(weight)])

            out = model(x)

            current = out['current'][:args.n_samples]
            current_loss = (metric(x_uniform[None], current[None], verts[None],
                                   faces[None], face_normals[None],
                                   bdry_edges[None])
                            .norm(p=2, dim=-1) * weight).sum() / weight.sum()

            f_inside = out['f'][args.n_samples:2*args.n_samples]
            f_outside = out['f'][2*args.n_samples:]
            io_loss = torch.maximum(f_outside - f_inside + 0.01,
                                    torch.zeros_like(f_inside)).mean()

            loss = current_loss + io_loss
            loss.backward()
            optimizer.step()

            writer.add_scalar('io_loss', io_loss.item(), it)
            writer.add_scalar('current_loss', current_loss.item(), it)
            writer.add_scalar('loss', loss.item(), it)
            pbar.set_postfix({
                'loss': loss.item(),
            })
            pbar.update(1)

            it += 1

            if it % 2000 == 0:
                scheduler.step()

            if it % 10000 == 0:
                torch.save({
                    'model': model.state_dict(),
                    'opt_state_dict': optimizer.state_dict(),
                    'sched_state_dict': scheduler.state_dict(),
                    'it': it,
                    'boundary': bdry_verts,
                    'mesh': data['fname']
                }, os.path.join(args.out, f'{it}.pth'))

            if it == args.n_iterations:
                done = True
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--out', type=str, default='out/reconstruction')
    parser.add_argument('--n_samples', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_iterations', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--rff_sigma', type=float, default=2)
    parser.add_argument('--metric', type=str,
                        default="anisotropic_reconstruction")

    args = parser.parse_args()
    main(args)
