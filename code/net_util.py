import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter, init
import torch_geometric
from torch_geometric.nn import graclus
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from torch_sparse import coalesce
from torch_scatter import scatter
import data_util


def batch_quat_to_rotmat(q, out=None, normalize=True):
    """transform quaternion to rotate matrix"""
    batchsize = q.size(0)

    if out is None:
        out = q.new_empty(batchsize, 3, 3)

    # 2 / squared quaternion 2-norm
    if normalize:
        s = 2/torch.sum(q.pow(2), 1)
    else:
        s = 2

    # coefficients of the Hamilton product of the quaternion with itself
    h = torch.bmm(q.unsqueeze(2), q.unsqueeze(1))

    out[:, 0, 0] = 1 - (h[:, 2, 2] + h[:, 3, 3]).mul(s)
    out[:, 0, 1] = (h[:, 1, 2] - h[:, 3, 0]).mul(s)
    out[:, 0, 2] = (h[:, 1, 3] + h[:, 2, 0]).mul(s)

    out[:, 1, 0] = (h[:, 1, 2] + h[:, 3, 0]).mul(s)
    out[:, 1, 1] = 1 - (h[:, 1, 1] + h[:, 3, 3]).mul(s)
    out[:, 1, 2] = (h[:, 2, 3] - h[:, 1, 0]).mul(s)

    out[:, 2, 0] = (h[:, 1, 3] - h[:, 2, 0]).mul(s)
    out[:, 2, 1] = (h[:, 2, 3] + h[:, 1, 0]).mul(s)
    out[:, 2, 2] = 1 - (h[:, 1, 1] + h[:, 2, 2]).mul(s)

    return out


def plot_wei(wei):
    import numpy as np
    import matplotlib.pyplot as plt
    y = np.sort(wei)
    x = np.arange(y.shape[0])
    plt.figure()
    plt.plot(x, y, color="red", linewidth=1)
    plt.show()
    pass


class PoolingLayer(torch.nn.Module):
    def __init__(self, in_channel, pool_type='max', pool_step=2, edge_weight_type=0, wei_param=2):
        super(PoolingLayer, self).__init__()
        assert (pool_type in ['max', 'mean'])
        self.pool_type = pool_type
        self.pool_step = pool_step
        self.edge_weight_type = edge_weight_type
        self.wei_param = wei_param

        if self.edge_weight_type in [4, 5]:
            self.lin = Linear(in_channel, in_channel)
        if self.edge_weight_type in [3, 4, 5]:
            # attention based edge weight for Graclus pooling
            self.att_l = Parameter(torch.Tensor(1, in_channel))
            self.att_r = Parameter(torch.Tensor(1, in_channel))
            init.xavier_uniform_(self.att_l.data, gain=1.414)
            init.xavier_uniform_(self.att_r.data, gain=1.414)

        self.unpooling_indices = None

    def forward(self, data, visual=False):
        val = data.edge_weight

        # yield data for Graclus pooling
        edge_weight = self._get_edge_weight(data)
        x, edge_index, pos = data.x, data.edge_index, data.pos
        edge_dual = data.edge_dual if hasattr(data, 'edge_dual') else None
        face = data.fv_indices if hasattr(data, 'fv_indices') else None

        # if visual and hasattr(data, 'fv_indices') and hasattr(data, 'name') and data.name[-1] == 'v':
        if visual:
            data_util.plot_graph(pos.cpu().numpy(), edge_index.T.cpu().numpy(), val.cpu().numpy())
            data_util.plot_graph(pos.cpu().numpy(), edge_index.T.cpu().numpy(), edge_weight.cpu().numpy())

            x_edge = x[edge_index]
            x_edge = ((x_edge[0] - x_edge[1])**2).sum(1)
            val = (x_edge/(-100)).exp()
            data_util.plot_graph(pos.cpu().numpy(), edge_index.T.cpu().numpy(), val.cpu().numpy())

            val = (x_edge/(-50)).exp()
            data_util.plot_graph(pos.cpu().numpy(), edge_index.T.cpu().numpy(), val.cpu().numpy())

            val = (x_edge/(-20)).exp()
            data_util.plot_graph(pos.cpu().numpy(), edge_index.T.cpu().numpy(), val.cpu().numpy())

            val = (x_edge/(-10)).exp()
            data_util.plot_graph(pos.cpu().numpy(), edge_index.T.cpu().numpy(), val.cpu().numpy())
            # plot_wei(val.cpu().numpy())

            # import matplotlib.pyplot as plt
            # import openmesh as om
            # c_map = plt.get_cmap('jet')
            # # val = ((data.x**2).sum(1)**0.5).cpu().numpy()
            # val = edge_weight.cpu().numpy()
            # val = val / val.max()
            # rgb = c_map(val)
            # points_12 = data.y[edge_index.T]
            # edge_len = ((points_12[:, 0] - points_12[:, 1])**2).sum(1)**0.5
            # points_3 = points_12.mean(1, keepdim=True) + edge_len.min()*0.05
            # points = torch.cat((points_12, points_3), dim=1)
            # points = points.reshape(-1, 3)
            # faces = torch.arange(points.shape[0], dtype=torch.long).reshape(-1, 3)
            # mesh = om.TriMesh(points.cpu().numpy(), faces.cpu().numpy())
            # for i in range(mesh.n_faces()):
            #     mesh.set_color(mesh.face_handle(i), rgb[i])
            # om.write_mesh(R"E:\SysFile\Desktop\visual_test.off", mesh, face_color=True)
            # pass

        # pooling loop steps
        clusts = []
        for _ in range(self.pool_step):
            cluster = graclus(edge_index, edge_weight, x.shape[0])
            cluster, perm = consecutive_cluster(cluster)
            clusts.append(cluster)

            if self.pool_type == 'mean':
                x = scatter(x, cluster, dim=0, reduce='mean')
            elif self.pool_type == 'max':
                x = scatter(x, cluster, dim=0, reduce='max')
            edge_index, edge_weight = pool_edge(cluster, edge_index, edge_weight)
            pos = None if pos is None else pool_pos(cluster, pos)
            edge_dual = None if edge_dual is None else cluster[edge_dual]
            # if group all nodes to single one, i.e. there is no edge
            if edge_index.numel() == 0:
                break

            visual = False
            if visual and face is not None:
                face = None if face is None else pool_face(cluster, face)
                import openmesh as om
                mesh = om.TriMesh(pos.cpu().numpy(), face.cpu().numpy())
                om.write_mesh(R"E:\SysFile\Desktop\pool_test.obj", mesh)
                pass



        # unpooling indices
        clust = clusts[-1]
        for c in clusts[-2::-1]:
            clust = clust[c]
        self.unpooling_indices = clust

        return torch_geometric.data.Data(x, edge_index, edge_dual=edge_dual, edge_weight=edge_weight, pos=pos, fv_indices=face)

    def _get_edge_weight(self, data):
        # remove self_loops
        edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None
        edge_index, edge_weight = remove_self_loops(data.edge_index, edge_weight)
        if edge_index.numel() == 0:
            return None
        data.edge_index = edge_index
        data.edge_weight = edge_weight

        if self.edge_weight_type == -1:  # None, random
            edge_weight = None
        elif self.edge_weight_type == 0:  # 0. normal and spatial difference (like bilateral filtering)
            edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None
        elif self.edge_weight_type == 1:
            feat_diff = data.x[edge_index]
            feat_diff = ((feat_diff[0] - feat_diff[1])**2).sum(1)
            edge_weight = (feat_diff/(-self.wei_param)).exp()
        elif self.edge_weight_type == 2:
            feat_diff = data.x[edge_index]
            feat_diff = ((feat_diff[0] - feat_diff[1])**2).sum(1)
            feat_diff = (feat_diff/(-self.wei_param)).exp()
            edge_weight = edge_weight * feat_diff
        elif self.edge_weight_type == 3:
            x = data.x
            # refer to the implementation of GATConv of 'pytorch_geometric'
            alpha = [(x * self.att_l).sum(dim=-1), (x * self.att_r).sum(dim=-1)]
            alpha_l = alpha[0][edge_index[0]] + alpha[1][edge_index[1]]
            alpha_r = alpha[0][edge_index[1]] + alpha[1][edge_index[0]]
            alpha = alpha_l + alpha_r
            edge_weight = F.sigmoid(alpha)
        elif self.edge_weight_type == 4:
            x = F.leaky_relu(self.lin(data.x), 0.2, inplace=True)
            # refer to the implementation of GATConv of 'pytorch_geometric'
            alpha = [(x * self.att_l).sum(dim=-1), (x * self.att_r).sum(dim=-1)]
            alpha_l = alpha[0][edge_index[0]] + alpha[1][edge_index[1]]
            alpha_r = alpha[0][edge_index[1]] + alpha[1][edge_index[0]]
            alpha = alpha_l + alpha_r
            edge_weight = F.sigmoid(alpha)
        elif self.edge_weight_type == 5:
            x = F.leaky_relu(self.lin(data.x), 0.2, inplace=True)
            # refer to the implementation of GATConv of 'pytorch_geometric'
            alpha = [(x * self.att_l).sum(dim=-1), (x * self.att_r).sum(dim=-1)]
            alpha_l = alpha[0][edge_index[0]] + alpha[1][edge_index[1]]
            alpha_r = alpha[0][edge_index[1]] + alpha[1][edge_index[0]]
            alpha = alpha_l + alpha_r
            wei = F.sigmoid(alpha)
            edge_weight = (wei + data.edge_weight) / 2
        elif self.edge_weight_type == 6:
            edge_weight = (edge_weight-edge_weight.min()) / (edge_weight.max()-edge_weight.min()+1e-12)
        elif self.edge_weight_type == 7:
            feat_diff = data.x[edge_index]
            feat_diff = ((feat_diff[0] - feat_diff[1])**2).sum(1)
            feat_diff = -feat_diff
            edge_weight = (feat_diff-feat_diff.min()) / (feat_diff.max()-feat_diff.min()+1e-12)
        elif self.edge_weight_type == 8:
            feat_diff = data.x[edge_index]
            feat_diff = ((feat_diff[0] - feat_diff[1])**2).sum(1)
            feat_diff = (feat_diff/(-2)).exp()
            edge_weight = (feat_diff-feat_diff.min()) / (feat_diff.max()-feat_diff.min()+1e-12)
        elif self.edge_weight_type == 9:
            feat_diff = data.x[edge_index]
            feat_diff = ((feat_diff[0] - feat_diff[1])**2).sum(1)
            feat_diff = (feat_diff/(-2)).exp()
            feat_diff = (feat_diff-feat_diff.min()) / (feat_diff.max()-feat_diff.min()+1e-12)
            edge_weight = (edge_weight-edge_weight.min()) / (edge_weight.max()-edge_weight.min()+1e-12)
            edge_weight = edge_weight + feat_diff
        elif self.edge_weight_type == 10:
            feat_diff = data.x[edge_index]
            feat_diff = ((feat_diff[0] - feat_diff[1])**2).sum(1)
            feat_diff = (feat_diff/(-2)).exp()
            edge_weight = edge_weight + feat_diff

        # for idx in range(data.x.shape[0]):
        #     edge_left = edge_index[0] == idx
        #     right_idx = edge_index[1][edge_left]
        #     wei_left = edge_weight[edge_left]
        #     edge_right = edge_index[1] == idx
        #     left_idx = edge_index[0][edge_right]
        #     wei_right = edge_weight[edge_right]

        return edge_weight

    def unpooling(self, x):
        if self.unpooling_indices is not None:
            x = x[self.unpooling_indices]
        return x


class DualFusionLayer(torch.nn.Module):
    def __init__(self, in_channel):
        super(DualFusionLayer, self).__init__()

        self.lin_v1 = Linear(in_channel*2, in_channel)
        self.lin_v2 = Linear(in_channel, in_channel)

        self.lin_f1 = Linear(in_channel*2, in_channel)
        self.lin_f2 = Linear(in_channel, in_channel)

    def forward(self, data_v, data_f):
        m = data_v.x.shape[0]
        n = data_f.x.shape[0]
        edge_dual = torch.stack([data_v.edge_dual, data_f.edge_dual], dim=0)
        if edge_dual.numel() > 0:
            edge_dual, _ = coalesce(edge_dual, None, m, n)

        x_v = self.fusion(data_v.x, edge_dual, data_f.x)
        x_f = self.fusion(data_f.x, edge_dual.flip(0), data_v.x)

        x_v = F.leaky_relu(self.lin_v1(x_v), 0.2, inplace=True)
        x_v = F.leaky_relu(self.lin_v2(x_v), 0.2, inplace=True)
        x_f = F.leaky_relu(self.lin_f1(x_f), 0.2, inplace=True)
        x_f = F.leaky_relu(self.lin_f2(x_f), 0.2, inplace=True)
        return x_v, x_f

    def fusion(self, x_i, edge_dual, x_j):
        row, col = edge_dual
        x_j = x_j[col]
        x_j = scatter(x_j, row, dim=0, reduce='mean')
        return torch.cat([x_i, x_j], dim=1)


class AdaptiveLossWeight(torch.nn.Module):
    def __init__(self, in_channel):
        pass

    def forward(self, feat_v, feat_f):
        return None


def pool_edge(cluster, edge_index, edge_attr=None, op="mean"):
    num_nodes = cluster.size(0)
    edge_index = cluster[edge_index.view(-1)].view(2, -1)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    if edge_index.numel() > 0:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes, op=op)
    return edge_index, edge_attr


def pool_face(cluster, fv_indices):
    face = cluster[fv_indices.view(-1)].view(-1, 3)
    invalid_flag = (face[:,0] == face[:,1]) | (face[:,0] == face[:,2]) | (face[:,1] == face[:,2])
    face = face[~invalid_flag]
    return face


def pooling(data, p_type='max', level=2, wei_type=0):
    clusts = []

    x, pos, edge_index = data.x, data.pos, data.edge_index

    if wei_type == 0:
        # 0. normal and spatial difference (like bilateral filtering)
        edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None
    elif wei_type == 1:
        # 1. L-2 norm of feature, Gaussian of norm difference
        edge_norm = (x**2).sum(1)
        edge_norm = edge_norm[edge_index]
        edge_weight = ((edge_norm[0]-edge_norm[1])**2 / (-2)).exp()
    elif wei_type == 2:
        # 2. Gaussian of L-2 norm of feature difference
        edge_norm = x[edge_index]
        edge_norm = ((edge_norm[0]-edge_norm[1])**2).sum(1)
        edge_weight = (edge_norm / (-2)).exp()

    for _ in range(level):
        cluster = graclus(edge_index, edge_weight, x.shape[0])
        cluster, perm = consecutive_cluster(cluster)
        clusts.append(cluster)

        if p_type == 'mean':
            x = scatter(x, cluster, dim=0, reduce='mean')
        elif p_type == 'max':
            x = scatter(x, cluster, dim=0, reduce='max')
        pos = None if pos is None else pool_pos(cluster, pos)
        edge_index, edge_weight = pool_edge(cluster, edge_index, edge_weight)
        # if group all nodes to single one, i.e. there is no edge
        if edge_index.numel() == 0:
            break

    # unpooling indices
    clust = clusts[-1]
    for c in clusts[-2::-1]:
        clust = clust[c]
    return torch_geometric.data.Data(x, edge_index, pos=pos, edge_weight=edge_weight), clust


def pooling_pre(data, step=2, level=2):
    edge_index = data.edge_index
    edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

    for i in range(1, level+1):
        clusters = []
        for _ in range(step):
            cluster = graclus(edge_index, edge_weight)
            cluster, perm = consecutive_cluster(cluster)
            clusters.append(cluster)
            edge_index, edge_weight = pool_edge(cluster, edge_index, edge_weight)

        # unpooling indices
        cluster_inv = clusters[-1]
        for c in clusters[-2::-1]:
            cluster_inv = cluster_inv[c]

        pool_info = {'clusters': clusters, 'cluster_inv': cluster_inv}
        setattr(data, F"pool_l{i}", pool_info)
    data.edge_weight = None
    return data


def pooling_run(data, pool_info, p_type='max'):
    clusters = pool_info['clusters']
    x, pos, edge_index = data.x, data.pos, data.edge_index
    for clust in clusters:
        if p_type == 'mean':
            x = scatter(x, clust, dim=0, reduce='mean')
        elif p_type == 'max':
            x = scatter(x, clust, dim=0, reduce='max')
        pos = None if pos is None else pool_pos(clust, pos)
        edge_index, _ = pool_edge(clust, edge_index)

    return torch_geometric.data.Data(x, edge_index, pos=pos)


if __name__ == "__main__":
    import os
    import glob
    import numpy as np
    import openmesh as om
    # import coarsening
    from datetime import datetime

    # from torch_geometric.nn import GATConv
    # x1 = torch.randn(4, 8)
    # edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    # conv = GATConv(8, 32, heads=1)
    # out = conv(x1, edge_index)

    dir_original = R'E:\code\python\denoising\TempNet\data\Synthetic\train\original'
    dir_noisy = R'E:\code\python\denoising\TempNet\data\Synthetic\train\noisy'

    dir_original = R'E:\code\python\denoising\TempNet\data\Synthetic\train'
    files_original = glob.glob(os.path.join(dir_original, "*.obj"))
    for filename in files_original:
        filename = R"E:\code\python\denoising\TempNet\data\Synthetic\train\Octahedron_n3.obj"

        mesh = om.read_trimesh(filename)
        # if mesh.n_faces() > 10000:
        #     continue

        points_v = mesh.points()
        fv_indices = mesh.fv_indices()
        vf_indices = mesh.vf_indices()
        points_f = points_v[fv_indices].mean(1)
        mesh.update_face_normals()
        face_n = mesh.face_normals()
        mesh.update_vertex_normals()
        vertex_n = mesh.vertex_normals()

        # vertex graph
        edge_idx_v = torch.from_numpy(mesh.edge_vertex_indices().T).long()  # directed grah, no self_loops
        edge_idx_v = to_undirected(edge_idx_v)
        edge_idx_v, _ = add_self_loops(edge_idx_v)
        edge_wei_v = data_util.calc_weight(torch.from_numpy(points_v), torch.from_numpy(vertex_n), edge_idx_v)
        graph_v = torch_geometric.data.Data(
            torch.from_numpy(points_v).float(), edge_idx_v,
            edge_weight=edge_wei_v.float(),
            pos=torch.from_numpy(points_v).float(),
            name="graph_v")

        # facet graph
        pos_f_n = torch.from_numpy(points_v[fv_indices].mean(1))
        edge_idx_f = data_util.build_facet_graph(torch.from_numpy(fv_indices), torch.from_numpy(vf_indices))
        edge_wei_f = data_util.calc_weight(pos_f_n, torch.from_numpy(face_n), edge_idx_f)
        graph_f = torch_geometric.data.Data(pos_f_n.float(), edge_idx_f, edge_weight=edge_wei_f.float(), name="graph_f")

        pool1 = PoolingLayer(3, edge_weight_type=0)
        data_r2 = pool1(graph_v)
        pool2 = PoolingLayer(3, edge_weight_type=0)
        data_r3 = pool2(data_r2)

        data_util.plot_graph(graph_v.pos.numpy(), graph_v.edge_index.numpy().T)
        data_util.plot_graph(data_r2.pos.numpy(), data_r2.edge_index.numpy().T)
        data_util.plot_graph(data_r3.pos.numpy(), data_r3.edge_index.numpy().T)


        pass


    print("--- END ---")
