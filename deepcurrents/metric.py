import torch

from . import utils


class Metric:
    @staticmethod
    def anisotropic_reconstruction(x, current, verts, faces,
                                   face_normals, bdry_edges):
        bs = x.shape[0]

        reshaped_faces = faces.view(bs, -1, 1).expand(-1, -1, 3)
        face_verts = torch.gather(verts, 1,
                                  reshaped_faces).view(-1, 3, 3)
        sq_dists, closest_faces = utils.point_face_distance(
            x.view(-1, 3),
            torch.tensor([x.shape[1] * i for i in range(bs)]).to(x.device),
            face_verts,
            torch.tensor([faces.shape[1] * i for i in range(bs)]).to(x.device),
            x.shape[1]
        )

        sq_dists_edges = utils.point_edge_distance(
            x.reshape(-1, 3),
            torch.tensor([x.shape[1] * i for i in range(bs)]).to(x.device),
            bdry_edges.view(-1, 2, 3),
            torch.tensor([bdry_edges.shape[1] * i
                          for i in range(bs)]).to(x.device),
            x.shape[1]
        )[0]
        bdry_idxs = (sq_dists - sq_dists_edges).abs() < 1e-5

        closest_normals = face_normals.flatten(0, 1)[closest_faces].view(
            bs, -1, 3)

        sq_dists = sq_dists[:, None]
        eps = torch.ones_like(sq_dists)
        eps[bdry_idxs] = 0
        eps = eps.view(bs, -1, 1)

        out = (current + (torch.sqrt(1-eps) - 1) *
               torch.einsum('abi,abj,abj->abi', closest_normals,
                            closest_normals, current))
        return out

    @staticmethod
    def uniform(x=None, current=None, verts=None, faces=None,
                face_normals=None, bdry_edges=None):
        return current
