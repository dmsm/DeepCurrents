import argparse

import imageio
import numpy as np
import torch

from deepcurrents import models, utils


def main(args):
    torch.manual_seed(1)
    np.random.seed(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = models.SurfaceModel()
    trained_state = torch.load(
        args.infile, map_location=device)
    model.load_state_dict(trained_state['model'])
    model.to(device)
    bdry = trained_state['boundary']

    def do_render(azimuth):
        return utils.render(model, args.resolution, args.min_depth,
                            args.max_depth, args.focal_length, azimuth,
                            elevation=args.elevation,
                            num_samples=args.num_samples,
                            num_presamples=args.num_presamples,
                            y_up=args.Y,
                            density_scale=args.density,
                            bdry=bdry)['current']

    with imageio.get_writer(args.outfile, mode='I') as writer:
        azimuths = np.linspace(0, 360, args.num_frames)[:-1]
        for az in azimuths:
            img = do_render(az)
            img = (255 * img).astype(np.uint8)
            writer.append_data(img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', type=str, default='surface.gif')
    parser.add_argument('--infile', type=str,
                        default='checkpoints/surface.pth')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--num_samples', type=int, default=128)
    parser.add_argument('--num_presamples', type=int, default=256)
    parser.add_argument('--min_depth', type=float, default=0.)
    parser.add_argument('--max_depth', type=float, default=np.sqrt(3))
    parser.add_argument('--elevation', type=float, default=0.)
    parser.add_argument('--focal_length', type=float, default=10.)
    parser.add_argument('--num_frames', type=int, default=40)
    parser.add_argument('-Y', action='store_true')
    parser.add_argument('--density', type=float, default=1e-1)

    args = parser.parse_args()
    main(args)
