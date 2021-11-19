import os

import igl
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import normalize
import torch


class MeshDataset(torch.utils.data.Dataset):

    def __init__(self, path, jitter=False, n_samples=1000, idx=None):
        fnames = sorted([os.path.join(path, f) for f in os.listdir(path)
                         if os.path.splitext(f)[1] in ('.obj', '.ply')])
        self.meshes = [(f, igl.read_triangle_mesh(f, dtypef=np.float32))
                       for f in fnames]
        self.jitter = jitter
        self.n_samples = n_samples
        self.idx = idx

        self.weights = []
        self.max_loop_size = 0
        for _, (verts, faces) in self.meshes:
            loops = igl.all_boundary_loop(faces)

            if jitter:
                bdry = np.zeros((verts.shape[0],), dtype=bool)
                bdry[np.concatenate(loops)] = True
                weight = np.zeros((verts.shape[0], bdry.sum()))
                L = igl.cotmatrix(verts, faces)
                int_weight = spsolve(
                    L[~bdry][:, ~bdry], -L[~bdry][:, bdry]).todense()
                weight[~bdry, :] = int_weight
                weight[np.concatenate(loops), :] = np.eye(bdry.sum())
                self.weights.append(weight)

            for loop in loops:
                self.max_loop_size = max(self.max_loop_size, len(loop))

    def __getitem__(self, index):
        if self.idx is not None:
            index = self.idx

        fname, (verts, faces) = self.meshes[index]

        loops = igl.all_boundary_loop(faces)
        if self.jitter:
            rot = R.from_euler('xyz', np.random.randint(-10, 11, size=3),
                               degrees=True).as_matrix()
            verts = verts @ rot.T

            displacements = []
            for i, loop in enumerate(loops):
                loop_verts = verts[loop]
                ev, P = np.linalg.eig(np.cov(loop_verts.T))
                P = P[:, ev.argsort()[::-1]]
                D = np.eye(3)
                D[0, 0] = np.random.rand()*0.3 + 0.85
                D[1, 1] = np.random.rand()*0.3 + 0.85
                loop_verts_ = loop_verts @ (P @ D @ P.T) + \
                    np.random.rand(3) * 0.05
                displacements.append(loop_verts_ - loop_verts)
            displacements = np.concatenate(displacements, axis=0)
            verts = verts + (self.weights[index] @ displacements)

            if verts[faces].max() > 1:
                verts = verts / verts[faces].max()

            max_shift = (
                1 - verts[faces].reshape(-1, 3)).min(0).clip(max=0.05)
            min_shift = -(
                verts[faces].reshape(-1, 3) + 1).min(0).clip(max=0.05)
            verts = verts + np.random.rand(3) * \
                (max_shift - min_shift) + min_shift

        verts = verts.astype(np.float32)

        face_normals = igl.per_face_normals(
            verts, faces, np.array([0, 0, 0], dtype=np.float32))
        vert_normals = igl.per_vertex_normals(verts, faces)

        # get boundary
        loop_lens = torch.tensor([len(loop) for loop in loops])
        loop_verts = [torch.tensor(verts[loop]) for loop in loops]
        loop_edges = torch.cat([loop[
            torch.from_numpy(np.roll(np.repeat(np.arange(
                loop.shape[0])[:, None], 2, axis=1).flatten(),
                                     -1).reshape(-1, 2))]
                                for loop in loop_verts], dim=0)

        # pad bdrys
        loop_verts = torch.stack(
            [torch.cat(
                [loop, loop[-1:].expand(self.max_loop_size-loop.shape[0], -1)],
                dim=0) for loop in loop_verts], dim=0)

        # sample points on target surface
        face_areas = igl.doublearea(verts, faces)
        bary = np.random.rand(self.n_samples, 3)
        bary /= bary.sum(axis=-1, keepdims=True)
        fidx = np.random.choice(face_areas.shape[0],
                                size=(self.n_samples,),
                                p=face_areas/face_areas.sum())
        x_surf = (bary[:, :, None] * verts[faces[fidx]]).sum(1)
        normals_surf = normalize(
            (bary[:, :, None] * vert_normals[faces[fidx]]).sum(1), axis=1)
        x_in = (x_surf - normals_surf *
                np.random.rand(self.n_samples, 1) * (0.02 - 0.001) + 0.001)
        x_out = (x_surf + normals_surf *
                 np.random.rand(self.n_samples, 1) * (0.02 - 0.001) + 0.001)

        return {
            'index': torch.tensor(index),
            'fname': fname,
            'verts': torch.tensor(verts),
            'vert_normals': torch.from_numpy(vert_normals),
            'faces': torch.tensor(faces),
            'face_normals': torch.from_numpy(face_normals),
            'bdry_verts': loop_verts,
            'bdry_edges': loop_edges,
            'bdry_lens': loop_lens,
            'x_outside': torch.from_numpy(x_out).float(),
            'x_inside': torch.from_numpy(x_in).float(),
        }

    def __len__(self):
        return len(self.meshes) if self.idx is None else 128


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
