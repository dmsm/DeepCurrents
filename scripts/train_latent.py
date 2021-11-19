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

    dataset = datasets.MeshDataset(args.data, n_samples=args.n_samples,
                                   jitter=True)
    dataloader = datasets.MultiEpochsDataLoader(
        dataset, batch_size=args.bs, shuffle=True, num_workers=8,
        drop_last=True)

    n_bdries = dataset[0]['bdry_verts'].shape[0]
    model = models.SurfaceModel(rff_sigma=args.rff_sigma, dim_z=args.dim_z,
                                dim_extra_z=args.dim_z*n_bdries,
                                n_data=len(dataset))

    model.to(device)
    model.train()

    bdry_encoder = models.BoundaryEncoder(dim_z=args.dim_z,
                                          n_boundaries=n_bdries)
    bdry_encoder.to(device)
    bdry_encoder.train()

    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': bdry_encoder.parameters()},
    ], lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                 gamma=args.sched_gamma)

    it = 0
    if args.preload is not None:
        ckpt = torch.load(args.preload, map_location=device)
        model.load_state_dict(ckpt['model'])
        bdry_encoder.load_state_dict(ckpt['bdry_encoder'])
        optimizer.load_state_dict(ckpt['opt_state_dict'])
        scheduler.load_state_dict(ckpt['sched_state_dict'])
        it = ckpt['it']

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    writer = SummaryWriter(os.path.join(
        args.out, datetime.datetime.now().strftime('train-%m%d%y-%H%M%S')),
        flush_secs=1)

    metric = getattr(Metric, args.metric)

    pbar = tqdm(total=args.n_iterations-it)
    done = False
    while not done:
        for batch in dataloader:
            optimizer.zero_grad()

            bdry_verts = batch['bdry_verts'].to(device)
            bdry_edges = batch['bdry_edges'].to(device)
            bdry_lens = batch['bdry_lens']
            verts = batch['verts'].to(device)
            faces = batch['faces'].to(device)
            face_normals = batch['face_normals'].to(device)
            x_inside = batch['x_inside'].to(device)
            x_outside = batch['x_outside'].to(device)

            x_uniform = torch.empty(
                args.bs, args.n_samples, 3).uniform_(-1, 1).to(device)
            x = torch.cat([x_uniform, x_inside, x_outside], dim=1)
            x.requires_grad = True

            with torch.no_grad():
                sq_dists_bdry = utils.point_edge_distance(
                    x_uniform[:, :args.n_samples//2].reshape(-1, 3),
                    torch.tensor([args.n_samples//2 * i
                                  for i in range(args.bs)]).to(device),
                    bdry_edges.view(-1, 2, 3),
                    torch.tensor([bdry_edges.shape[1] * i
                                  for i in range(args.bs)]).to(device),
                    args.n_samples
                )[0].view(args.bs, -1)
                weight = torch.exp(-sq_dists_bdry / (2 * 0.1**2))
                weight = torch.cat([weight, torch.ones_like(weight)], dim=1)

            z = bdry_encoder([bdry[:len] for bdry, len in
                              zip(bdry_verts.unbind(1), bdry_lens[0])])
            latent_z = model.embedding(batch['index'].to(device))
            z = torch.cat([z, latent_z], dim=1)[:, None].expand(-1,
                                                                x.shape[1], -1)
            out = model(x.flatten(0, 1), z=z.flatten(0, 1), bdry=bdry_verts)

            current = out['current'].view(args.bs, -1, 3)[:, :args.n_samples]
            f = out['f'].view(args.bs, -1, 1)

            current_loss = (metric(x_uniform, current, verts, faces,
                                   face_normals, bdry_edges)
                            .norm(p=2, dim=-1) * weight).sum() / weight.sum()

            f_inside = f[:, args.n_samples:2*args.n_samples]
            f_outside = f[:, 2*args.n_samples:]
            io_loss = torch.maximum(f_outside - f_inside + 0.01,
                                    torch.zeros_like(f_inside)).mean()

            loss = current_loss + io_loss
            loss.backward()
            optimizer.step()

            pbar.set_postfix({
                'loss': loss.item(),
            })
            pbar.update(1)
            writer.add_scalar('io_loss', io_loss.item(), it)
            writer.add_scalar('current_loss', current_loss.item(), it)
            writer.add_scalar('loss', loss.item(), it)

            it += 1

            if it % args.sched_interval == 0:
                scheduler.step()

            if it % 10000 == 0:
                torch.save({
                    'model': model.state_dict(),
                    'bdry_encoder': bdry_encoder.state_dict(),
                    'opt_state_dict': optimizer.state_dict(),
                    'sched_state_dict': scheduler.state_dict(),
                    'it': it,
                    'n_data': len(dataset),
                    'n_bdries': n_bdries
                }, os.path.join(args.out, f'{it}.pth'))

            if it == args.n_iterations:
                done = True
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--out', type=str, default='out/latent')
    parser.add_argument('--preload', type=str, default=None)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--n_samples', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--sched_gamma', type=float, default=0.5)
    parser.add_argument('--sched_interval', type=int, default=60000)
    parser.add_argument('--n_iterations', type=int, default=300000)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--rff_sigma', type=float, default=2)
    parser.add_argument('--dim_z', type=int, default=256)
    parser.add_argument('--metric', type=str,
                        default="anisotropic_reconstruction")

    args = parser.parse_args()
    main(args)
