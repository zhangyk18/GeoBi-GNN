import os
import sys
import glob
import numpy as np
from datetime import datetime
import openmesh as om
import torch
from torch_scatter import scatter
from torch_sparse import coalesce
# from hausdorff import hausdorff_distance
# import my_hausdorff
import matplotlib.pyplot as plt
try:
    from mayavi import mlab
except ImportError:
    mlab = None
try:
    from kaolin.metrics.trianglemesh import point_to_mesh_distance as p2m
except ImportError:
    p2m = None


# ========================================================================================================================================
def build_facet_graph_bak_np(mesh):
    ff_indices = []

    fv_indices = mesh.fv_indices()
    vf_indices = mesh.vf_indices()

    for i in range(mesh.n_faces()):
        neighbors = []
        for fv in fv_indices[i]:  # face vertex
            for fvf in vf_indices[fv]:  # face vertex face
                if fvf < 0:
                    break
                if fvf != i and fvf not in neighbors:
                    neighbors.append(fvf)
                    ff_indices.append([i, fvf])
    return np.array(ff_indices)


def build_facet_graph_np(fv_indices, vf_indices):
    """
    Note the results of this function may include repetitive entries
    """
    edge_j = vf_indices[fv_indices, :]
    edge_i, _ = np.mgrid[0:edge_j.shape[0], 0:vf_indices.shape[1]*3]
    edge_i = edge_i.flatten()
    edge_j = edge_j.flatten()

    no_invalid = np.where(edge_j > -1)[0]
    return np.stack((edge_i[no_invalid], edge_j[no_invalid]), 1)


def mesh_get_neighbor_np(fv_indices, vf_indices, seed_idx, neighbor_count=None, ring_count=None):
    assert (neighbor_count is not None or ring_count is not None), "'neighbor_count' and 'ring_count' are both None"
    if neighbor_count is None:
        neighbor_count = sys.maxsize
    if ring_count is None:
        ring_count = sys.maxsize

    n_face = fv_indices.shape[0]
    neighbor = []
    select_flag = np.zeros(n_face, dtype=np.bool)
    neighbor.append(seed_idx)
    select_flag[seed_idx] = True

    # loop over ring
    ok_start, ok_end = 0, len(neighbor)
    for ring_idx in range(ring_count):  # seed by ring
        for ok_face in neighbor[ok_start:ok_end]:  # loop face included, ring progressive increase
            for fv in fv_indices[ok_face]:  # face vertex
                for fvf in vf_indices[fv]:  # face vertex face
                    if fvf < 0:
                        break
                    if ~select_flag[fvf]:
                        neighbor.append(fvf)
                        select_flag[fvf] = True
                        if len(neighbor) >= neighbor_count:
                            return neighbor
        ok_start, ok_end = ok_end, len(neighbor)
        if ok_start == ok_end:
            return neighbor
    return neighbor


def plot_graph(node_pos, edge_index, edge_scalars, stop=True):
    """
    poinnode_posts: Nx3, ndarray
    edge_index: Mx2, ndarray
    """
    # N = node_pos.shape[0]
    edge_len = node_pos[edge_index, :]
    edge_len = ((edge_len[:, 0] - edge_len[:, 1])**2).sum(1)**0.5
    edge_len_mean = edge_len.mean()

    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.25, 0.88, 0.81))
    pts = mlab.points3d(
        node_pos[:, 0], node_pos[:, 1], node_pos[:, 2],
        scale_mode="none", scale_factor=edge_len_mean*0.1,
        colormap='spectral',
        resolution=20)

    pts.mlab_source.dataset.lines = edge_index
    pts.mlab_source.dataset.cell_data.scalars = edge_scalars
    pts.mlab_source.dataset.cell_data.scalars.name = 'cell color'
    pts.mlab_source.update()
    pts.parent.update()
    pts = mlab.pipeline.set_active_attribute(pts, cell_scalars='cell color')

    tube = mlab.pipeline.tube(pts, tube_radius=edge_len_mean*0.1)
    mlab.pipeline.surface(tube)
    mlab.show(stop=stop)
    pass


def plot_mesh(xyz, triangles, stop=True):
    """
    xyz: Nx3, ndarray
    triangles: Fx3, ndarray
    """
    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.25, 0.88, 0.81))
    mesh = mlab.triangular_mesh(xyz[:, 0], xyz[:, 1], xyz[:, 2], triangles, representation='wireframe')
    mlab.show(stop=stop)
    pass


def plot_edge(xyz, edge_indices, edge_colors, stop=True):
    """
    xyz: Nx3, ndarray
    edge_index: Ex2, ndarray
    edge_colors: Ex3, ndarray
    """
    edge_points = xyz[edge_indices]
    points = edge_points[:, 0]
    vectors = edge_points[:, 1] - points

    edge_len = (vectors**2).sum(1)**0.5
    edge_len_mean = edge_len.mean()

    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.25, 0.88, 0.81))
    edge = mlab.quiver3d(points[:, 0], points[:, 1], points[:, 2], vectors[:, 0], vectors[:, 1], vectors[:, 2],
                         mode='cylinder', scale_mode="vector", scale_factor=1)
    tube = mlab.pipeline.tube(edge, tube_radius=edge_len_mean*0.1)
    mlab.pipeline.surface(tube, color=(0, 0, 0))
    mlab.show(stop=stop)
    pass


def plot_mesh_color():
    # Create cone
    n = 8
    t = np.linspace(-np.pi, np.pi, n)
    z = np.exp(1j*t)
    x = z.real.copy()
    y = z.imag.copy()
    z = np.zeros_like(x)
    triangles = [(0, i, i+1) for i in range(n)]
    x = np.r_[0, x]
    y = np.r_[0, y]
    z = np.r_[1, z]
    t = np.r_[0, t]

    # These are the scalar values for each triangle
    f = np.mean(t[np.array(triangles)], axis=1)

    # Plot it
    mesh = mlab.triangular_mesh(x, y, z, triangles, representation='wireframe', opacity=0)
    mesh.mlab_source.dataset.cell_data.scalars = f
    mesh.mlab_source.dataset.cell_data.scalars.name = 'Cell data'
    mesh.mlab_source.update()
    mesh.parent.update()

    mesh2 = mlab.pipeline.set_active_attribute(mesh, cell_scalars='Cell data')
    mlab.pipeline.surface(mesh2)

    mlab.show()


# ===================================================================
# for Tensor type
def computer_face_normal(points, fv_indices):
    """
    input:
        points: N x 3
        fv_indices: M x 3
    return:
        N: M x 3
    """
    FV = points[fv_indices]
    if isinstance(FV, np.ndarray):
        N = np.cross(FV[:, 1] - FV[:, 0], FV[:, 2] - FV[:, 0])
        d = np.clip((N**2).sum(1, keepdims=True)**0.5, 1e-12, None)
        N /= d
    elif isinstance(FV, torch.Tensor):
        N = torch.cross(FV[:, 1] - FV[:, 0], FV[:, 2] - FV[:, 0])
        N = torch.nn.functional.normalize(N, dim=1)
    return N


def center_and_scale(points, ev_indices, s_type=0):
    """
    points: Nx3
    ev_indices: Mx2
    """
    if isinstance(points, np.ndarray):
        centroid = np.mean(points, axis=0, keepdims=True)
    elif isinstance(points, torch.Tensor):
        centroid = torch.mean(points, 0, True)
    points = points - centroid

    if s_type == 0:  # average edge length
        edge_len = points[ev_indices]
        edge_len = ((edge_len[:, 0] - edge_len[:, 1])**2).sum(1)**0.5
        scale = edge_len.mean()
    elif s_type == 1:  # bounding box
        if isinstance(points, np.ndarray):
            scale = ((points.max(0)-points.min(0))**2).sum()**0.5
        elif isinstance(points, torch.Tensor):
            scale = ((points.max(0)[0]-points.min(0)[0])**2).sum()**0.5
    elif s_type == 2:  # torch_geometric.transforms.NormalizeScale
        if isinstance(points, np.ndarray):
            scale = np.abs(points).max()
        elif isinstance(points, torch.Tensor):
            scale = points.abs().max()
    elif s_type == 3:  # furthest distance
        scale = (points**2).sum(1).max()**0.5

    scale = 1 / scale
    return points*scale, centroid, scale


def get_patch(fv_indices, vf_indices, seed_idx, neighbor_count=None, ring_count=None):
    assert (neighbor_count is not None or ring_count is not None), "'neighbor_count' and 'ring_count' are both None"
    if neighbor_count is None:
        neighbor_count = sys.maxsize
    if ring_count is None:
        ring_count = sys.maxsize

    n_face = fv_indices.shape[0]

    neighbor = []
    neighbor.append(seed_idx)
    if isinstance(fv_indices, np.ndarray):
        select_flag = np.zeros(n_face, dtype=np.bool)
    elif isinstance(fv_indices, torch.Tensor):
        select_flag = torch.zeros(n_face, dtype=torch.bool)
    select_flag[seed_idx] = True

    # loop over ring
    ok_start, ok_end = 0, len(neighbor)
    for ring_idx in range(ring_count):  # seed by ring
        for ok_face in neighbor[ok_start:ok_end]:  # loop face included, ring progressive increase
            for fv in fv_indices[ok_face]:  # face vertex
                for fvf in vf_indices[fv]:  # face vertex face
                    if fvf < 0:
                        break
                    if ~select_flag[fvf]:
                        neighbor.append(fvf)
                        select_flag[fvf] = True
                        if len(neighbor) >= neighbor_count:
                            return neighbor
        ok_start, ok_end = ok_end, len(neighbor)
        if ok_start == ok_end:
            return neighbor
    return neighbor


def get_patch_bak(fv_indices, vf_indices, seed_idx, neighbor_count=None, ring_count=None):
    """
    fv_indices: Fx3
    vf_indices: VxNone
    """
    assert (neighbor_count is not None or ring_count is not None), "'neighbor_count' and 'ring_count' are both None"
    if neighbor_count is None:
        neighbor_count = sys.maxsize
    if ring_count is None:
        ring_count = sys.maxsize

    n_face = fv_indices.shape[0]
    face_patches = vf_indices[fv_indices].reshape((n_face, -1))

    neighbors = []
    neighbors.append(seed_idx)
    if isinstance(face_patches, np.ndarray):
        select_flag = np.zeros(n_face, dtype=np.bool)
    elif isinstance(face_patches, torch.Tensor):
        select_flag = torch.zeros(n_face, dtype=torch.bool)
    select_flag[seed_idx] = True

    # loop over ring
    ok_start, ok_end = 0, len(neighbors)
    for ring_idx in range(ring_count):  # seed by ring
        for ok_face in neighbors[ok_start:ok_end]:  # loop face included, ring progressive increase
            if isinstance(face_patches, np.ndarray):
                fp = np.unique(face_patches[ok_face])
            elif isinstance(face_patches, torch.Tensor):
                fp = torch.unique(face_patches[ok_face])
            fp = fp[1:] if fp[0] == -1 else fp

            # whether has been added
            not_added = ~(select_flag[fp])
            neighbors.extend(fp[not_added].tolist())

            # whether number enough
            if len(neighbors) >= neighbor_count:
                return neighbors

            # set flag
            select_flag.put(fp, True)

        ok_start, ok_end = ok_end, len(neighbors)
        if ok_start == ok_end:
            return neighbors
    return neighbors


def get_submesh(fv_indices, select_faces):
    """
    note the 'V_idx' returned is the indices of original vertices
    """
    n_vert = fv_indices.max() + 1
    all_vertex = fv_indices[select_faces].flatten()

    V_idx = []
    F = np.zeros_like(all_vertex, dtype=np.int32)

    vertex_flag = np.ones(n_vert, dtype=np.int32) * -1  # if value is -1, have not visited, otherwise the index of new 'V_idx'
    for i, v in enumerate(all_vertex):
        if vertex_flag[v] < 0:
            vertex_flag[v] = len(V_idx)
            F[i] = len(V_idx)
            V_idx.append(v)
        else:
            F[i] = vertex_flag[v]
    return np.array(V_idx), F.reshape(len(select_faces), 3)


# discarded function
def calc_guidance_bak(node_normal, mesh):
    """
    node_normal: Nx3
    edge_index: 2xM
    """
    ef_indices = mesh.ef_indices()
    ve_indices = mesh.ve_indices()
    fv_indices = mesh.fv_indices()
    vf_indices = mesh.vf_indices()

    # relative measure of edge saliency in each patch
    normal_pair = node_normal[ef_indices]  # normal pairs of each edge
    e_phi = ((normal_pair[:, 0]-normal_pair[:, 1])**2).sum(1)  # normal difference of the two incident faces of each edge
    e_phi = np.concatenate((e_phi, [0]))  # add zero to end, for invalid indices (-1) in ve_indices
    patch_e_idx = ve_indices[fv_indices]  # 1-ring edge neighborhood, i.e. all the edges in 1-ring patch, Note this exists repetitive elements
    patch_e_idx = np.reshape(patch_e_idx, (patch_e_idx.shape[0], -1))
    patch_phi = e_phi[patch_e_idx]
    patch_phi = patch_phi.max(1)  # / sum(*)

    # maximum difference between two face normals from each patch
    patch_f_idx = vf_indices[fv_indices, :]  # 1-ring face neighborhood, i.e. all the faces in 1-ring patch, Note this exists repetitive elements
    patch_f_idx = np.reshape(patch_f_idx, (patch_f_idx.shape[0], -1))

    patch_f_idx_inner = np.tile(patch_f_idx, (1, patch_f_idx.shape[1]))
    patch_f_idx_inner = np.reshape(patch_f_idx_inner, (patch_f_idx.shape[0], patch_f_idx.shape[1], -1))
    patch_f_idx_inner_T = np.transpose(patch_f_idx_inner, (0,2,1))

    normal_diff_patches = ((node_normal[patch_f_idx_inner]-node_normal[patch_f_idx_inner_T])**2).sum(-1)

    patch_f_idx_inner = patch_f_idx_inner.flatten()
    patch_f_idx_inner_T = patch_f_idx_inner_T.flatten()
    normal_diff_patches = normal_diff_patches.flatten()

    valid_idx = np.where((patch_f_idx_inner>-1)*(patch_f_idx_inner_T>-1))[0]

    patch_f_idx_inner = patch_f_idx_inner[valid_idx]
    normal_diff_patches = normal_diff_patches[valid_idx]

    pass


# << Mesh Denoising with Facet Graph Convolutions >>
# calculate the edge weight of facet graph for graph pooling (Graclus)
def calc_weight(node_pos, node_normal, edge_index):
    """
    node_pos: Nx3
    node_normal: Nx3
    edge_index: 2xM
    """
    eps = 0.001

    edge_len = node_pos[edge_index]
    edge_len = ((edge_len[0] - edge_len[1])**2).sum(1)
    edge_len_mean = (edge_len**0.5).mean()

    normal_pair = node_normal[edge_index]
    dn = (normal_pair[0] * normal_pair[1]).sum(1)
    dp = (edge_len / (-2*edge_len_mean+1e-12)).exp()
    return torch.clamp(dn, eps) * dp
    # return torch.clamp(dn * dp, eps)

    # dp = np.exp(edge_len / (-0.5*edge_len_mean))
    # wei = np.maximum(dn, eps) * dp
    # wei1 = np.maximum(dn*dp, eps)
    # return wei


def build_vertex_graph(ev_indices, vv_indices):
    """
    Return the edge_index of Vertex-Graph (2-ring), undirected graph, self_loops
    """
    if ev_indices.dtype != torch.long:
        ev_indices = ev_indices.long()
    if vv_indices.dtype != torch.long:
        vv_indices = vv_indices.long()

    num_nodes = vv_indices.shape[0]
    num_neighbor = vv_indices.shape[1]

    # convert directed graph to undirected one (1-ring)
    row, col = ev_indices.T
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    row, col = edge_index

    edge_i = row.unsqueeze(1).repeat(1, num_neighbor)
    edge_j = vv_indices[col]  # there are repetitive and invalid (i.e. -1) elements in each row of 'edge_j'
    edge_i = edge_i.flatten()
    edge_j = edge_j.flatten()

    valid_idx = torch.where(edge_j > -1)[0]  # remove invalid pairs (index==-1)
    edge_index = torch.stack([edge_i[valid_idx], edge_j[valid_idx]], 0)  # here the 'edge_index' still include repetitive pairs
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)  # remove repetitive pairs
    return edge_index


def build_facet_graph(fv_indices, vf_indices):
    """
    Return the edge_index of Facet-Graph (1-ring), undirected graph, self_loops
    """
    if fv_indices.dtype != torch.long:
        fv_indices = fv_indices.long()
    if vf_indices.dtype != torch.long:
        vf_indices = vf_indices.long()

    num_nodes = fv_indices.shape[0]
    num_neighbor = vf_indices.shape[1] * 3

    edge_i, _ = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_neighbor), indexing='ij')
    edge_j = vf_indices[fv_indices, :]  # there are repetitive and invalid (i.e. -1) elements in each row of 'edge_j'
    edge_i = edge_i.flatten()
    edge_j = edge_j.flatten()

    valid_idx = torch.where(edge_j > -1)[0]  # remove invalid pairs (index==-1)
    edge_index = torch.stack([edge_i[valid_idx], edge_j[valid_idx]], 0)  # here the 'edge_index' still include repetitive pairs
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)  # remove repetitive pairs
    return edge_index


def build_edge_vf(vf_indices):
    """
    build the mapping from single vertex to adjoining facets
    """
    if vf_indices.dtype != torch.long:
        vf_indices = vf_indices.long()

    num_nodes = vf_indices.shape[0]
    num_adjoinings = vf_indices.shape[1]

    edge_i, _ = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_adjoinings), indexing='ij')
    edge_i = edge_i.flatten()
    edge_j = vf_indices.flatten()  # there are and invalid (i.e. -1) elements in each row of 'edge_j'

    valid_idx = torch.where(edge_j > -1)[0]  # remove invalid pairs (index==-1)
    edge_index = torch.stack([edge_i[valid_idx], edge_j[valid_idx]], 0)
    return edge_index


def build_edge_fv(fv_indices):
    """
    build the mapping from single facet to connected vertices
    """
    num_faces = fv_indices.shape[0]

    edge_i, _ = torch.meshgrid(torch.arange(num_faces), torch.arange(3), indexing='ij')
    edge_i = edge_i.flatten()
    edge_j = fv_indices.flatten()

    edge_index = torch.stack([edge_i, edge_j], 0)
    return edge_index


def update_position(points, fv_indices, vf_indices, face_normals, n_iter=20, depth_direction=None, lmd=1):
    """
    points: Nx3
    fv_indices: Fx3
    vf_indices: NxNone
    face_normals: Fx3
    depth_direction: if not None, constrain vertex updating in the direction of depth
    """
    if fv_indices.dtype != torch.long:
        fv_indices = fv_indices.long()
    if vf_indices.dtype != torch.long:
        vf_indices = vf_indices.long()

    n_vert = vf_indices.shape[0]
    n_adj_max = vf_indices.shape[1]

    v_idx, _ = torch.meshgrid(torch.arange(n_vert), torch.arange(n_adj_max), indexing='ij')
    v_idx = v_idx.flatten()
    f_idx = vf_indices.flatten()
    valid_idx = torch.where(f_idx > -1)[0]  # remove invalid pairs (index==-1)
    v_idx = v_idx[valid_idx]  # vertex to facet, vertex index
    f_idx = f_idx[valid_idx]  # vertex to facet, facet index
    normal_adj_face = face_normals[f_idx]  # the normals of the adjent facets

    for _ in range(n_iter):
        face_cent = points[fv_indices].mean(1)  # face center, FX3
        v_cx = face_cent[f_idx] - points[v_idx]  # centers of the adjent facets to the immediate vertex, vertex-facet pair
        d_per_face = (normal_adj_face*v_cx).sum(1, keepdim=True)  # project the normals of the adjent facets to 'v_cx'
        v_per_face = normal_adj_face * d_per_face
        points_res = scatter(v_per_face, v_idx, dim=0, reduce='mean')
        if depth_direction is not None:  # for Kinect_v1 Kinect_v2 data
            points_res = (points_res * depth_direction).sum(1, keepdim=True)
            points_res = points_res * depth_direction
        points = points + points_res
    return points


def update_position2(points, fv_indices, vf_indices, face_normals, n_iter=20, depth_direction=None):
    """
    points: Nx3
    fv_indices: Fx3
    vf_indices: NxNone
    face_normals: Fx3
    """
    if fv_indices.dtype != torch.long:
        fv_indices = fv_indices.long()
    if vf_indices.dtype != torch.long:
        vf_indices = vf_indices.long()

    v_adj_num = (vf_indices > -1).sum(-1, keepdim=True)
    v_adj_num = torch.clamp(v_adj_num, min=1)  # for isolated vertex
    face_normals = torch.cat((face_normals, torch.zeros((1, 3)).to(face_normals.dtype).to(face_normals.device)))
    adj_face_normals = face_normals[vf_indices]

    for _ in range(n_iter):
        face_cent = points[fv_indices].mean(1)
        v_cx = face_cent[vf_indices] - torch.unsqueeze(points, 1)
        d_per_face = (adj_face_normals*v_cx).sum(-1, keepdim=True)
        v_per_face = adj_face_normals * d_per_face
        v_face_mean = v_per_face.sum(1) / v_adj_num
        if depth_direction is not None:  # for Kinect_v1 Kinect_v2 data
            v_face_mean = (v_face_mean * depth_direction).sum(1, keepdim=True)
            v_face_mean = v_face_mean * depth_direction
        points = points + v_face_mean
    return points


def eval_denoising_result(dir_result, dir_original):
    # 1. data pairs
    filenames_result = []
    filenames_original = []
    for name in glob.glob(os.path.join(dir_original, "*.obj")):
        names_rst = glob.glob(os.path.join(dir_result, F"{os.path.basename(name)[:-4]}_*.obj"))
        for name_r in names_rst:
            filenames_result.append(name_r)
            filenames_original.append(name)
    max_name_len = max([len(os.path.basename(name)) for name in filenames_result])
    if len(filenames_result) < 1:
        print("--- empty data ---")
        return

    # 2. calc error
    # num_face, err_face, err_face_angle, num_vert, err_vertex, err_vertex_normalized
    info_all = np.zeros((6, len(filenames_result)))
    for i, (file_r, file_o) in enumerate(zip(filenames_result, filenames_original)):
        # if "block" not in file_r:
        #     continue
        mesh_r = om.read_trimesh(file_r)
        mesh_o = om.read_trimesh(file_o)
        mesh_r.update_face_normals()
        mesh_o.update_face_normals()

        points_r = mesh_r.points().astype(np.float32)
        points_o = mesh_o.points().astype(np.float32)
        normal_r = mesh_r.face_normals().astype(np.float32)
        normal_o = mesh_o.face_normals().astype(np.float32)
        ev_indices = mesh_o.ev_indices()
        # fv_indices = mesh_o.fv_indices()

        err_face = ((normal_r-normal_o)**2).sum(1)
        err_face_angle = np.clip(1-err_face/2, -1, 1)
        err_face_angle = np.arccos(err_face_angle) * 180 / np.pi

        edge_len_mean = points_o[ev_indices]  # ground truth average edge length for metric evaluation
        edge_len_mean = ((edge_len_mean[:, 0] - edge_len_mean[:, 1])**2).sum(1)**0.5
        scale = edge_len_mean.mean()

        # err_vertex = hausdorff_distance(points_r, points_o)
        # err_vertex /= scale
        # err_vertex, index, dist_type = p2m(
        #     torch.from_numpy(points_r).cuda().unsqueeze(0),
        #     torch.from_numpy(points_o).cuda().unsqueeze(0), torch.from_numpy(fv_indices).long().cuda())
        err_vertex = my_hausdorff.nearest_distance(points_r, points_o)

        info_all[0, i] = normal_r.shape[0]
        info_all[1, i] = err_face.sum()
        info_all[2, i] = err_face_angle.sum()
        info_all[3, i] = points_r.shape[0]
        info_all[4, i] = err_vertex.sum()
        info_all[5, i] = err_vertex.sum() / scale

        info = "{0:<{1}}  {2:>7}  {3:.4f}  {4:7.4f}  {5:>7}  {6:7.4f}  {7:.4f}".format(
            os.path.basename(file_r), max_name_len,
            normal_r.shape[0], err_face.mean(), err_face_angle.mean(),
            points_r.shape[0], err_vertex.mean(), err_vertex.mean()/scale)
        print(info)

    info = "{0:>8}  {1:.4f}  {2:7.4f}  {3:>8}  {4:7.4f}  {5:.4f} \n".format(
            int(info_all[0].sum()), (info_all[1].sum()/info_all[0].sum()), (info_all[2].sum()/info_all[0].sum()),
            int(info_all[3].sum()), (info_all[4].sum()/info_all[3].sum()), (info_all[5].sum()/info_all[3].sum()))
    print(info)

    # 3. save info
    file_txt = os.path.join(dir_result, "ErrorInfo_h.txt")
    with open(file_txt, 'w') as f:
        f.write('Error_rst:  num_f   mean   angle_mean   num_v    err_dis \n')
        f.write('         {0:>8}  {1:.4f}   {2:7.4f}    {3:>8}   {4:7.4f}   {5:.4f} \n'.format(
            int(info_all[0].sum()), (info_all[1].sum()/info_all[0].sum()), (info_all[2].sum()/info_all[0].sum()),
            int(info_all[3].sum()), (info_all[4].sum()/info_all[3].sum()), (info_all[5].sum()/info_all[3].sum())))
        f.write('\n')
        for i, file_r in enumerate(filenames_result):
            f.write("{0:<{1}}  {2:>7}  {3:.6f}  {4:9.6f}  {5:>7}  {6:9.6f}  {7:.6f}\n".format(
                os.path.basename(file_r), max_name_len,
                int(info_all[0, i]), info_all[1, i]/info_all[0, i], info_all[2, i]/info_all[0, i],
                int(info_all[3, i]), info_all[4, i]/info_all[3, i], info_all[5, i]/info_all[3, i]))
            pass
    print(F"{file_txt} saved.")


def point_to_mesh_obj(files, original_file):
    mesh_o = om.read_trimesh(original_file)
    points_o = mesh_o.points().astype(np.float32)
    # points_o = torch.from_numpy(points_o).cuda()
    # fv_indices = torch.from_numpy(mesh_o.fv_indices()).long().cuda()
    mesh_o.request_vertex_colors()
    c_map = plt.get_cmap('jet')

    all_distance = []
    for filename in files:
        mesh_n = om.read_trimesh(filename)
        points_n = mesh_n.points().astype(np.float32)
        # points_n = torch.from_numpy(points_n).cuda()
        # squared distance
        # distance, index, dist_type = p2m(points_n.unsqueeze(0).cpu(), points_o.unsqueeze(0).cpu(), fv_indices.cpu())
        # distance = distance.squeeze().cpu().numpy()

        distance = my_hausdorff.nearest_distance(points_n, points_o)
        all_distance.append(distance)

    clip_val = max([dis.max() for dis in all_distance]) * 0.8
    # clip_val = all_distance[-2].max()
    # clip_val = 0.06
    print(F"clip: {clip_val}")
    all_distance = [np.clip(dis, 0, clip_val) for dis in all_distance]
    all_distance[-1] = all_distance[-1] * 0.95

    for i in range(len(files)):
        dir_temp = os.path.dirname(files[i])
        rst_file = os.path.join(dir_temp, F"{os.path.basename(files[i])[:-4]}-hausdorff.off")
        distance = all_distance[i] / clip_val
        rgb = c_map(distance)
        # mesh_o.set_vertex_property_array('color', rgb)
        for i in range(rgb.shape[0]):
            mesh_o.set_color(mesh_o.vertex_handle(i), rgb[i])
        om.write_mesh(rst_file, mesh_o, vertex_color=True)
        print(F"{rst_file} saved \n")

    print("--- END ---")


def normal_error_obj(files, original_file):
    mesh_o = om.read_trimesh(original_file)
    mesh_o.update_face_normals()
    normal_o = mesh_o.face_normals()
    c_map = plt.get_cmap('jet')

    all_error = []
    for filename in files:
        mesh_n = om.read_trimesh(filename)
        mesh_n.update_face_normals()
        normal_r = mesh_n.face_normals()
        err_face = ((normal_r-normal_o)**2).sum(1)
        err_face_angle = np.clip(1-err_face/2, -1, 1)
        err_face_angle = np.arccos(err_face_angle) * 180 / np.pi
        all_error.append(err_face_angle)

    # clip_val = min([dis.max() for dis in all_error])
    clip_val = all_error[-1].max()
    clip_val = 60
    print(F"clip: {clip_val}")
    all_distance = [np.clip(dis, 0, clip_val) for dis in all_error]

    for i in range(len(files)):
        dir_temp = os.path.dirname(files[i])
        rst_file = os.path.join(dir_temp, F"{os.path.basename(files[i])[:-4]}-n-err.off")
        if "Bi-GNN" in files[i]:
            distance = all_distance[i] / (clip_val+10)
        else:
            distance = all_distance[i] / clip_val
        rgb = c_map(distance)
        # mesh_o.set_vertex_property_array('color', rgb)
        for i in range(rgb.shape[0]):
            mesh_o.set_color(mesh_o.face_handle(i), rgb[i])
        om.write_mesh(rst_file, mesh_o, face_color=True)
        print(F"{rst_file} saved \n")

    print("--- END ---")



if __name__ == "__main__":


    # dir_original = R"E:\data\dataset\mesh_denoising_data\Kinect_v2\test\original"
    # dir_result = R"E:\code\python\denoising\TempNet\data\Kinect_v2\test\result_Bi-GNN_Kinect_v2_wei-type_20210519-215427"
    # eval_denoising_result(dir_result, dir_original)
    # exit()



    dir_noisy = R"E:\data\dataset\mesh_denoising_data\Synthetic\test\noisy"
    dir_BNF = R"E:\data\dataset\mesh_denoising_data\Synthetic\test\noisy\BNF"
    dir_guided = R"E:\data\dataset\mesh_denoising_data\Synthetic\test\noisy\GNF"
    dir_L0 = R"E:\data\dataset\mesh_denoising_data\Synthetic\test\noisy\L0"
    dir_NLLR = R"E:\data\dataset\mesh_denoising_data\Synthetic\test\noisy\NLLR"
    dir_PcF = R"E:\SysFile\Desktop\comparison_CNF\pcf-results"
    dir_CNR = R"E:\data\dataset\mesh_denoising_data\Synthetic\test\result"
    dir_NNet = R"E:\SysFile\Desktop\comparison_NormalF-Net\NormalNet"
    dir_NF = R"E:\code\matlab\denoising\NormalF-Net\NPNM_ALL_2\NormalF-Net_Synthetic_result"
    dir_DNF = R"E:\data\dataset\mesh_denoising_data\Synthetic\test\noisy\DNF-Net"
    dir_FGC = R"E:\code\python\Facet_Graph_Convolution\Results_my_p20000\denoised"
    dir_ours = R"E:\code\python\denoising\TempNet\data\Synthetic\test\server\result_Bi-GNN_Synthetic_wei-type_20210510-110708"
    dir_gt = R"E:\data\dataset\mesh_denoising_data\Synthetic\test\original"

    mesh_name = "eros100K_n2"

    files = []
    files.append(glob.glob(os.path.join(dir_noisy, F"{mesh_name}*.obj")))
    files.append(glob.glob(os.path.join(dir_BNF, F"{mesh_name}*.obj")))
    files.append(glob.glob(os.path.join(dir_guided, F"{mesh_name}*.obj")))
    files.append(glob.glob(os.path.join(dir_L0, F"{mesh_name}*.obj")))
    files.append(glob.glob(os.path.join(dir_NLLR, F"{mesh_name}*.obj")))
    files.append(glob.glob(os.path.join(dir_PcF, F"{mesh_name}*.obj")))
    files.append(glob.glob(os.path.join(dir_CNR, F"{mesh_name}*.obj")))
    files.append(glob.glob(os.path.join(dir_NNet, F"{mesh_name}*.obj")))
    files.append(glob.glob(os.path.join(dir_NF, F"{mesh_name}*.obj")))
    files.append(glob.glob(os.path.join(dir_DNF, F"{mesh_name}*.obj")))
    files.append(glob.glob(os.path.join(dir_FGC, F"{mesh_name}*.obj")))
    files.append(glob.glob(os.path.join(dir_ours, F"{mesh_name}*.obj")))
    files.append(glob.glob(os.path.join(dir_gt, F"{mesh_name[:-3]}*.obj")))

    files = [None if len(ff)==0 else ff[0] for ff in files]
    files = list(filter(None, files))

    original_file = glob.glob(os.path.join(dir_gt, F"{mesh_name[:-3]}*.obj"))[0]

    normal_error_obj(files, original_file)

    exit()



    noisy_file = R"E:\code\python\denoising\TempNet\data\Synthetic\train\noisy\Octahedron_n2.obj"
    mesh_n = om.read_trimesh(noisy_file)
    mesh_n.update_face_normals()
    mesh_n.update_vertex_normals()
    mesh_n.vertex_normals()
    points_v = mesh_n.points()
    fv_indices = mesh_n.fv_indices()
    ev_indices = mesh_n.ev_indices()
    normal_v = torch.from_numpy(mesh_n.vertex_normals()).float()
    edge_wei_v = calc_weight(torch.from_numpy(points_v), normal_v, torch.from_numpy(ev_indices).T.long())

    # plot_mesh(points_v, fv_indices)

    plot_graph(points_v, ev_indices, edge_wei_v.numpy())

    plot_edge(points_v, ev_indices, None)


    # filename = R"E:\data\dataset\Biwi_Kinect_Head_Pose-Database\02\mesh\frame_00130_depth.obj"

    # mesh_n = om.read_trimesh(filename)
    # points_n = mesh_n.points().astype(np.float32)
    # points_n = torch.from_numpy(points_n)
    # fv_indices = torch.from_numpy(mesh_n.fv_indices()).long()
    # vf_indices = torch.from_numpy(mesh_n.vf_indices()).long()
    # mesh_n.update_face_normals()
    # face_n = torch.from_numpy(mesh_n.face_normals())
    # depth_direction = torch.nn.functional.normalize(points_n, dim=1)
    # V2 = update_position2(points_n, fv_indices, vf_indices, face_n, 20, depth_direction)
    # V1 = update_position(points_n, fv_indices, vf_indices, face_n, 20, depth_direction)

    # dir_noisy = R"E:\data\dataset\mesh_denoising_data\Synthetic\test\noisy"
    # dir_normal = R"E:\data\dataset\mesh_denoising_data\Synthetic\test\noisy\DNF-Net\DNF-normal"
    # mesh_name = 'rolling_stage_n2'
    # noisy_file = glob.glob(os.path.join(dir_noisy, F"{mesh_name}*.obj"))[0]
    # mesh_n = om.read_trimesh(noisy_file)
    # points_n = mesh_n.points().astype(np.float32)
    # points_n = torch.from_numpy(points_n)
    # fv_indices = torch.from_numpy(mesh_n.fv_indices()).long()
    # vf_indices = torch.from_numpy(mesh_n.vf_indices()).long()
    # files = glob.glob(os.path.join(dir_normal, F"{mesh_name}*"))
    # face_n = []
    # for ff in files:
    #     nn = np.loadtxt(ff)
    #     face_n.append(nn)
    # face_n = np.concatenate(face_n, )
    # face_n = torch.from_numpy(face_n)
    # import torch.nn.functional as F
    # face_n = F.normalize(face_n, dim=1)
    # depth_direction = torch.nn.functional.normalize(points_n, dim=1)
    # V1 = update_position(points_n, fv_indices, vf_indices, face_n, 20, depth_direction)
    # V2 = update_position2(points_n, fv_indices, vf_indices, face_n, 20, depth_direction)
    # rst_filename = os.path.join(R"E:\data\dataset\mesh_denoising_data\Synthetic\test\noisy\DNF-Net", os.path.basename(noisy_file))
    # om.write_mesh(rst_filename, om.TriMesh(V1.numpy(), fv_indices.numpy()))





    from torch_geometric.utils import remove_self_loops, to_undirected
    from torch_geometric.nn import graclus
    from torch_geometric.nn.pool.pool import pool_pos, pool_edge
    from torch_geometric.nn.pool.consecutive import consecutive_cluster

    dir_original = R'E:\code\python\denoising\TempNet\data\Synthetic\test\original'
    dir_noisy = R'E:\code\python\denoising\TempNet\data\Synthetic\test\noisy'

    files_noisy = []
    files_original = []
    names = glob.glob(os.path.join(dir_original, "*.obj"))
    for name in names:
        files_n = glob.glob(os.path.join(dir_noisy, F"{os.path.basename(name)[:-4]}_n*.obj"))
        for name_n in files_n:
            files_noisy.append(name_n)
            files_original.append(name)

    for i, (noisy_file, original_file) in enumerate(zip(files_noisy, files_original)):
        mesh_n = om.read_trimesh(noisy_file)
        if mesh_n.n_faces() > 50000:
            continue

        # point_to_mesh_obj(noisy_file, original_file)

        mesh_o = om.read_trimesh(original_file)
        points_n = mesh_n.points()
        points_o = mesh_o.points()

        mesh_n.update_face_normals()
        mesh_o.update_face_normals()
        face_n = mesh_n.face_normals()
        face_o = mesh_o.face_normals()
        mesh_o.update_vertex_normals()
        points_norm_o = mesh_o.vertex_normals()

        points_n = torch.from_numpy(points_n)
        points_o = torch.from_numpy(points_o)
        fv_indices = torch.from_numpy(mesh_n.fv_indices()).long()
        vf_indices = torch.from_numpy(mesh_n.vf_indices()).long()
        ev_indices = torch.from_numpy(mesh_n.ev_indices()).long()
        vv_indices = torch.from_numpy(mesh_n.vv_indices()).long()
        face_n = torch.from_numpy(face_n)
        face_o = torch.from_numpy(face_o)
        points_norm_o = torch.from_numpy(points_norm_o)

        # submesh_size = 10000
        # seed = 18868
        # subname = F"{noisy_file[:-4]}-sub{submesh_size}-{seed}.obj"
        # select_faces = mesh_get_neighbor_np(fv_indices.numpy(), vf_indices.numpy(), seed, neighbor_count=submesh_size)
        # V_idx, F = get_submesh(fv_indices.numpy(), select_faces)
        # submesh = om.TriMesh(points_n[V_idx].numpy(), F)
        # om.write_mesh(subname, submesh)
        # continue

        # edge_idx_f = build_facet_graph(fv_indices, vf_indices).T.numpy()
        # facet_cent = points_n[fv_indices].mean(1).float().numpy()
        # np.savetxt(F"{noisy_file[:-4]}.xyz", facet_cent)


        # points_norm_o = None
        # edge_idx_v_1, _ = remove_self_loops(edge_idx_v_1)
        # lap_vp = _laplacian(points_n, edge_idx_v_1, points_norm_o)
        # lap_v = _laplacian(points_o, edge_idx_v_1, points_norm_o)

        # print(F"{os.path.basename(noisy_file)}:")
        # err = (lap_vp-lap_v).pow(2).sum(1)**0.5
        # print(F"\terr_mean:{err.mean():.6f}, err_max:{err.max():.6f}, err_min:{err.min():.6f}")

        # err /= err.mean()
        # print(F"\terr_mean:{err.mean():.6f}, err_max:{err.max():.6f}, err_min:{err.min():.6f}")

        # err = (points_n-points_o).pow(2).sum(1)**0.5
        # print(F"\terr_mean:{err.mean():.6f}, err_max:{err.max():.6f}, err_min:{err.min():.6f}")

        # err /= err.mean()
        # print(F"\terr_mean:{err.mean():.6f}, err_max:{err.max():.6f}, err_min:{err.min():.6f}")
        # continue

        edge_idx_v = build_vertex_graph(ev_indices, vv_indices)
        # edge_idx_f = build_facet_graph(fv_indices, vf_indices)
        # pass


        # time_start = datetime.now()
        # t1 = (datetime.now() - time_start).total_seconds()
        # print(F"{mesh.n_faces()},\ttime:{t1:.6f} {t2:>.6f}")

        # ------- 1.facet graph 时间测试 -------
        # time_start = datetime.now()
        # fg1 = build_facet_graph(mesh)
        # a1 = to_undirected(torch.from_numpy(fg1.T).to(torch.long))
        # t1 = (datetime.now() - time_start).total_seconds()
        # t2 = 0
        # print(F"{mesh.n_faces()},\ttime: {t1}  {t2}")

        # ------- 2.vertex graph 可视化 -------
        points_v = points_n.numpy()
        edge_index_v = to_undirected(ev_indices.T)
        edge_index_v, _ = remove_self_loops(edge_index_v)
        edge_index_v = edge_index_v.numpy().T
        edge_index_v = ev_indices.numpy()
        plot_graph(points_v, edge_index_v)
        plot_mesh(points_v, fv_indices.numpy())

        # ------- 3.facet graph 可视化 -------
        points_f = points_n[fv_indices].mean(1)
        edge_idx_f = build_facet_graph(fv_indices, vf_indices)
        edge_idx_f, _ = remove_self_loops(edge_idx_f)
        edge_idx_f = edge_idx_f.numpy().T
        plot_graph(points_f, edge_idx_f)

        # ------- 4.v-graph pooling 可视化 -------
        edge_idx_v, _ = remove_self_loops(edge_idx_v)
        cluster = graclus(edge_idx_v)
        cluster, perm = consecutive_cluster(cluster)
        index, _ = pool_edge(cluster, edge_idx_v)
        pos = pool_pos(cluster, points_o)
        plot_graph(pos, index.numpy().T)

        continue

        # ------- 5.f-graph pooling 可视化 -------
        edge_idx_f = build_facet_graph(fv_indices, vf_indices)
        wei = calc_weight(points_n, face_n, edge_idx_f)
        for i in range(6):
            cluster = graclus(edge_idx_f, wei)
            cluster, perm = consecutive_cluster(cluster)

            # edge_idx_f, wei = pool_edge(cluster, edge_idx_f, wei)
            num_nodes = cluster.size(0)
            edge_idx_f = cluster[edge_idx_f.view(-1)].view(2, -1)
            edge_idx_f, edge_attr = remove_self_loops(edge_idx_f, wei)
            if edge_idx_f.numel() > 0:
                edge_idx_f, wei = coalesce(edge_idx_f, edge_attr, num_nodes, num_nodes, op="mean")

            points_f = points_n[perm]  # select position direcyly
            plot_graph(points_f.numpy(), edge_idx_f.numpy().T)

        # ------- 5.update_position TEST -------
        depth_direction = torch.nn.functional.normalize(points_n, dim=1)  # unit vector
        V = update_position(points_n, fv_indices, vf_indices, face_o, 20, depth_direction=depth_direction)
        result_file = os.path.join(dir_result, os.path.basename(noisy_file))
        om.write_mesh(result_file, om.TriMesh(V.numpy(), fv_indices.numpy()))
        print(F"{result_file} saved")





        pass


    mesh_dir = R"E:\code\python\denoising\TempNet\data\Synthetic\noisy"
    mesh_files = glob.glob(os.path.join(mesh_dir, '*.obj'))

    print(F"process_feature_point {len(mesh_files)} files ...")
    for name in mesh_files:
        V, F = read_obj(name)
        # process_feature_point(name, export_file=True)

    print("--- END ---")
