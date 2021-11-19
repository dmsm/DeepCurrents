import argparse

import imageio
import numpy as np
import torch

from deepcurrents import datasets, models, utils


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trained_state = torch.load(
        args.infile, map_location=device)

    dataset = datasets.MeshDataset(path=args.data)
    n_bdries = dataset[0]['bdry_verts'].shape[0]
    model = models.SurfaceModel(rff_sigma=args.rff_sigma, dim_z=args.dim_z,
                                dim_extra_z=args.dim_z*n_bdries,
                                n_data=len(dataset))
    model.to(device)
    model.eval()
    model.load_state_dict(trained_state['model'])

    bdry_encoder = models.BoundaryEncoder(dim_z=args.dim_z,
                                          n_boundaries=n_bdries)
    bdry_encoder.to(device)
    bdry_encoder.eval()
    bdry_encoder.load_state_dict(trained_state['bdry_encoder'])

    all_bdry_verts = []
    for i in range(args.num_interpolants):
        all_bdry_verts.append(dataset[i]['bdry_verts'].to(device))

    latent_zs = model.embedding(torch.from_numpy(
        np.random.choice(len(dataset), args.num_interpolants,
                         replace=False)).to(device))
    bdry_lens = dataset[0]['bdry_lens']

    def do_render(bdry_verts, z):
        return utils.render(model, args.resolution, args.min_depth,
                            args.max_depth, args.focal_length,
                            elevation=10, azimuth=45,
                            num_samples=args.num_samples,
                            num_presamples=args.num_presamples, z=z,
                            bdry=bdry_verts, density_scale=0.05)['current']

    ts = list(np.linspace(0, 1, args.num_frames))
    with imageio.get_writer(args.outfile, mode='I') as writer:
        for i, j in [(x, (x+1) % args.num_interpolants)
                     for x in range(args.num_interpolants)]:
            if args.interpolation_type == 'boundary':
                latent_z = latent_zs[0]
                for t in ts:
                    bdry_verts = t * \
                        all_bdry_verts[j] + (1-t) * all_bdry_verts[i]
                    bdry_z = bdry_encoder([bdry[None, :len] for bdry, len in
                                           zip(bdry_verts, bdry_lens)]
                                          ).squeeze(0)
                    z = torch.cat([bdry_z, latent_z])

                    img = do_render(bdry_verts, z)
                    img = np.round(255 * img).astype(np.uint8)
                    writer.append_data(img)
            elif args.interpolation_type == 'latent':
                bdry_verts = all_bdry_verts[0]
                bdry_z = bdry_encoder([bdry[None, :len] for bdry, len in
                                       zip(bdry_verts, bdry_lens)]
                                      ).squeeze(0)
                for t in ts:
                    latent_z = t * latent_zs[j] + (1-t) * latent_zs[i]
                    z = torch.cat([bdry_z, latent_z])

                    img = do_render(bdry_verts, z)
                    img = np.round(255 * img).astype(np.uint8)
                    writer.append_data(img)
            else:
                raise ValueError('Invalid interpolation type.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', type=str, default='surface.gif')
    parser.add_argument('--infile', type=str,
                        default='checkpoints/surface.pth')
    parser.add_argument('--data', type=str)
    parser.add_argument('--resolution', type=int, default=200)
    parser.add_argument('--num_interpolants', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--num_presamples', type=int, default=500)
    parser.add_argument('--min_depth', type=float, default=0.)
    parser.add_argument('--max_depth', type=float, default=np.sqrt(3))
    parser.add_argument('--focal_length', type=float, default=10.)
    parser.add_argument('--num_frames', type=int, default=5)
    parser.add_argument('--dim_z', type=int, default=256)
    parser.add_argument('--rff_sigma', type=float, default=2)
    parser.add_argument('--interpolation_type', type=str, default='latent')
    parser.add_argument('--seed', type=int, default=5)

    args = parser.parse_args()
    main(args)
