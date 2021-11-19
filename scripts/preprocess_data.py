import argparse
import os

import igl
import numpy as np
import tqdm


def main(args):
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    fnames = sorted([
        os.path.join(args.meshes, f) for f in os.listdir(args.meshes)
        if os.path.splitext(f)[1] in ('.obj', '.ply')])
    seg_ids = np.loadtxt(args.seg)

    scale = 0
    for f in fnames:
        verts, faces = igl.read_triangle_mesh(fnames[0])
        faces = faces[seg_ids == args.seg_idx]
        cc = igl.face_components(faces)
        faces = faces[cc == 0]
        verts = verts - verts[faces].mean((0, 1))
        scale = max(scale, 2 * np.absolute(verts[faces]).max())

    if not args.skip_alignment:
        source_verts, faces = igl.read_triangle_mesh(fnames[0])

        # select segment
        faces = faces[seg_ids == args.seg_idx]
        cc = igl.face_components(faces)
        faces = faces[cc == 0]

        # normalize data
        source_verts = (source_verts -
                        source_verts[faces].mean(axis=(0, 1))) / scale

    for f in tqdm.tqdm(fnames):
        verts, faces = igl.read_triangle_mesh(f)

        # select segment
        faces = faces[seg_ids == args.seg_idx]
        cc = igl.face_components(faces)
        faces = faces[cc == 0]

        # normalize data
        verts = (verts - verts[faces].mean((0, 1))) / scale

        if not args.skip_alignment:
            H = verts[faces].reshape(-1, 3).T \
                @ source_verts[faces].reshape(-1, 3)
            U, S, V = np.linalg.svd(H)
            R = V.T @ U.T
            verts = verts @ R.T

        verts, faces, _ = igl.faces_first(verts, faces)
        verts = verts[:faces.max()+1]
        igl.write_triangle_mesh(os.path.join(args.out, os.path.basename(f)),
                                verts, faces)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--meshes', type=str,
        default='data/human_benchmark_sig17/meshes/train/faust/')
    parser.add_argument(
        '--seg', type=str,
        default=
        'data/human_benchmark_sig17/segs/train/faust/faust_corrected.txt')
    parser.add_argument('--out', type=str)
    parser.add_argument('--seg_idx', type=int, default=2)
    parser.add_argument('--skip_alignment', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
