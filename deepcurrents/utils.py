import gc
import math

import numpy as np
from pytorch3d import _C
from scipy.spatial.transform import Rotation as Rot
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def biot_savart_3d(x, verts):
    x = x[:, None, :, None]  # [n_bdrys, 1, m, 1, 3]
    verts = verts[:, :, None]  # [n_bdrys, num_verts, 1, n, 3]
    edges = torch.roll(verts, -1, dims=-2) - verts
    tangents = F.normalize(edges, p=2, dim=-1)
    disp = x - verts
    dir0 = F.normalize(disp, p=2, dim=-1)
    dir1 = torch.roll(dir0, -1, dims=-2)
    normals = torch.cross(tangents.expand(-1, -1, x.shape[2], -1, -1),
                          disp, dim=-1)
    perpdist = normals.norm(p=2, dim=-1, keepdim=True)
    dot0 = (dir0 * tangents).sum(-1, keepdim=True)
    dot1 = (dir1 * tangents).sum(-1, keepdim=True)
    V = normals * (dot1 - dot0) / perpdist.pow(2).clamp(min=1e-10)
    return V.sum((1, 3)) / 1000


def render(model, resolution, min_depth=0, max_depth=math.sqrt(3),
           focal_length=10, azimuth=45, elevation=0,
           num_samples=128, num_presamples=256, y_up=True,
           density_scale=1e-1,
           keys=['current'], **render_args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create pixel grid in image coordinates
    pixels = torch.stack(torch.meshgrid(
        [torch.linspace(-0.51, 0.51, resolution, device=device)] * 2), dim=-1)
    pixels = torch.cat([pixels, torch.zeros(
        resolution, resolution, 1, device=device)], dim=-1)
    pixels = pixels.flatten(0, 1)[:, None]

    # Camera information
    cam_offset = (min_depth + max_depth) / 2
    image_center = torch.tensor(
        [0, 0, cam_offset]).float().to(device)[None, None]
    pinhole = torch.tensor(
        [0, 0, cam_offset + focal_length]).float().to(device)[None, None]
    if y_up:
        camera_matrix = torch.Tensor(
            Rot.from_euler('zxy', [-90, -elevation, azimuth],
                           degrees=True).as_matrix()
        ).float().to(device)
    else:
        camera_matrix = torch.Tensor(
            Rot.from_euler('yz', [90 - elevation, azimuth],
                           degrees=True).as_matrix()
        ).float().to(device)

    # Convert to world coordinates
    pixels = (pixels + image_center) @ camera_matrix.T
    pinhole = pinhole @ camera_matrix.T

    imgsize = resolution**2
    batchsize = 2**12 // num_samples
    img = {key: [] for key in keys}
    with tqdm(total=imgsize // batchsize, desc="Rendering") as pbar:
        for pixel_batch in torch.chunk(pixels, imgsize // batchsize):
            actual_batchsize = pixel_batch.shape[0]
            rays = F.normalize(pixel_batch - pinhole, p=2, dim=-1)

            # Get initial samples and compute interval widths
            front = (min_depth * rays + pixel_batch).clamp(-1, 1)
            rear = (max_depth * rays + pixel_batch).clamp(-1, 1)
            num_intervals = num_presamples - 1
            interval_widths = (front - rear).norm(p=2, dim=-1) / num_intervals
            t = torch.linspace(0, 1, num_presamples, device=device)[
                None, :, None]
            presamples = ((1 - t) * front + t * rear)  # .flatten(0, 1)

            with torch.no_grad():
                # Evaluate "CDF" to compute adaptive samples
                fs = model(presamples.flatten(0, 1), only_f=True, **render_args
                           ).view(actual_batchsize, num_presamples)
                interval_prob = fs.diff(dim=1).abs()
                interval_prob = interval_prob / \
                    interval_prob.sum(dim=1, keepdim=True)
                sample_interval_idx = torch.multinomial(
                    interval_prob, num_samples, replacement=True
                ).sort(dim=1).values
                sample_linear_idx = (
                    torch.arange(actual_batchsize, device=device).view(
                        actual_batchsize, 1) *
                    num_intervals + sample_interval_idx)
                sample_weight = (
                    interval_widths /
                    interval_prob.view(
                        actual_batchsize * num_intervals
                    )[sample_linear_idx]).view(actual_batchsize,
                                               num_samples, 1)
                adapted_depth = (min_depth + interval_widths *
                                 (sample_interval_idx +
                                  torch.empty(sample_interval_idx.shape,
                                              device=device).uniform_(0, 1))
                                 ).sort(dim=1).values[:, :, None]
                samples = (adapted_depth * rays + pixel_batch)

            # Evaluate current
            samples.requires_grad = True
            raw = model(samples.flatten(0, 1), **render_args)

            with torch.no_grad():
                for key in keys:
                    # Convert raw density values to alpha values
                    raw[key] = raw[key].detach().view(
                        actual_batchsize, num_samples, 3)
                    density = raw[key].norm(dim=-1, p=2, keepdim=True)
                    alpha = 1 - torch.exp(-density_scale *
                                          density * sample_weight)

                    # Compositing weights are propabilities of passing from
                    # camera to sample without reflecting, then reflecting at
                    # sample
                    composite_weight = (1 - alpha).log().cumsum(dim=-2).exp()
                    composite_weight = composite_weight.roll(shifts=1, dims=-2)
                    composite_weight[:, 0, :] = 1
                    composite_weight = composite_weight * alpha

                    # Convert directions to colors
                    hue = (0.5 + 0.5 * (raw[key] / density))
                    hue = 1 - hue / hue.max(dim=-1, keepdim=True).values

                    # Composite colors with white background to get final image
                    img[key].append(((composite_weight * hue).sum(dim=1)
                                     + (1 - composite_weight.sum(dim=1))
                                     ).detach())

            pbar.update(1)

    for key in keys:
        img[key] = normalize_image(
            torch.cat(img[key], dim=0)
        ).view(resolution, resolution, 3).detach().cpu().numpy()

    return img


def normalize_image(x):
    x /= x.max()
    return x


class InputMapping(nn.Module):
    def __init__(self, d_in, d_out, sigma=2):
        super().__init__()
        self.B = nn.Parameter(torch.randn(d_out // 2, d_in) * sigma,
                              requires_grad=False)

    def forward(self, x):
        x = (2*np.pi*x) @ self.B.T
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class _PointFaceDistance(Function):
    """
    Torch autograd Function wrapper PointFaceDistance Cuda implementation
    """

    @staticmethod
    def forward(ctx, points, points_first_idx, tris, tris_first_idx,
                max_points):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first
                point index in each example in the batch
            tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The
                `t`-th triangular face is spanned by `(tris[t, 0], tris[t, 1],
                tris[t, 2])`
            tris_first_idx: LongTensor of shape `(N,)` indicating the first
                face index in each example in the batch
            max_points: Scalar equal to maximum number of points in the batch
        Returns:
            dists: FloatTensor of shape `(P,)`, where `dists[p]` is the squared
                euclidean distance of `p`-th point to the closest triangular
                face in the corresponding example in the batch
            idxs: LongTensor of shape `(P,)` indicating the closest triangular
                face in the corresponding example in the batch.

            `dists[p]` is `d(points[p], tris[idxs[p], 0], tris[idxs[p], 1],
            tris[idxs[p], 2])` where `d(u, v0, v1, v2)` is the distance of
            point `u` from the triangular face `(v0, v1, v2)`

        """
        dists, idxs = _C.point_face_dist_forward(
            points, points_first_idx, tris, tris_first_idx, max_points
        )
        ctx.save_for_backward(points, tris, idxs)
        ctx.mark_non_differentiable(idxs)
        return dists, idxs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists, grad_idxs):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        grad_points, grad_tris = _C.point_face_dist_backward(
            points, tris, idxs, grad_dists
        )
        return grad_points, None, grad_tris, None, None


point_face_distance = _PointFaceDistance.apply


class _PointEdgeDistance(Function):
    """
    Torch autograd Function wrapper PointEdgeDistance Cuda implementation
    """

    @staticmethod
    def forward(ctx, points, points_first_idx, segms, segms_first_idx,
                max_points):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first
                point index for each example in the mesh
            segms: FloatTensor of shape `(S, 2, 3)` of edge segments. The
                `s`-th edge segment is spanned by `(segms[s, 0], segms[s, 1])`
            segms_first_idx: LongTensor of shape `(N,)` indicating the first
                edge index for each example in the mesh
            max_points: Scalar equal to maximum number of points in the batch
        Returns:
            dists: FloatTensor of shape `(P,)`, where `dists[p]` is the squared
                euclidean distance of `p`-th point to the closest edge in the
                corresponding example in the batch
            idxs: LongTensor of shape `(P,)` indicating the closest edge in the
                corresponding example in the batch.

            `dists[p] = d(points[p], segms[idxs[p], 0], segms[idxs[p], 1])`,
            where `d(u, v0, v1)` is the distance of point `u` from the edge
            segment spanned by `(v0, v1)`.
        """
        dists, idxs = _C.point_edge_dist_forward(
            points, points_first_idx, segms, segms_first_idx, max_points
        )
        ctx.save_for_backward(points, segms, idxs)
        ctx.mark_non_differentiable(idxs)
        return dists, idxs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists, grad_idxs):
        grad_dists = grad_dists.contiguous()
        points, segms, idxs = ctx.saved_tensors
        grad_points, grad_segms = _C.point_edge_dist_backward(
            points, segms, idxs, grad_dists
        )
        return grad_points, None, grad_segms, None, None


point_edge_distance = _PointEdgeDistance.apply
