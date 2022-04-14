import os
import math
import torch
from torch import nn
import torch.utils.data
import torch.nn.parallel
import torch.nn.functional as F
import torch_geometric
from torch_scatter import scatter
from torch_geometric.nn import GCNConv, FeaStConv
from torch_geometric.utils import remove_self_loops
import data_util
from net_util import GACConv, pooling, pooling_run, PoolingLayer, AdaptiveLossWeight, batch_quat_to_rotmat, DualFusionLayer
from torch_geometric.nn import GATConv
from kaolin.metrics.pointcloud import sided_distance
from kaolin.metrics.pointcloud import chamfer_distance
try:
    from pytorch3d.ops.points_alignment import iterative_closest_point as icp
except ImportError:
    icp = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'log')


class FacetAttentionGNN(torch.nn.Module):
    def __init__(self, input_feat, bn=True, bias=True):
        super(FacetAttentionGNN, self).__init__()

        self.gcn1 = GCNConv(3, 16)
        # self.gcn2 = GCNConv(16, 16)
        self.gcn3 = GCNConv(16, 32)

        self.mlp_global = nn.Sequential(
            # nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
        )

        self.mlp_feat = nn.Sequential(
            nn.Linear(128+32, 128), nn.BatchNorm1d(128), nn.ReLU(),
            # nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 32), nn.BatchNorm1d(32), nn.ReLU(),
        )

        self.mlp_diff = nn.Sequential(
            # nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(),
        )

        self.mlp_atten = nn.Sequential(
            nn.Linear(2, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 1), nn.BatchNorm1d(1),  # nn.ReLU(),
        )

        # self.gcn4 = GCNConv(32, 32)
        self.gcn5 = GCNConv(32, 32)
        self.gcn6 = GCNConv(32, 128)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, dual_data):
        data = dual_data[1]
        x, edge_index = data.x, data.edge_index
        x = x[:, 3:]
        xyz = x[:, :3]

        feat = self.gcn1(x, edge_index).relu()
        # feat = self.gcn2(feat, edge_index, xyz).relu()
        feat = self.gcn3(feat, edge_index).relu()

        feat_global = self.mlp_global(feat)
        if hasattr(data, 'batch'):
            feat_global = torch_geometric.nn.global_max_pool(feat_global, data.batch)
            feat_global = feat_global.gather(0, data.batch.unsqueeze(1).expand(-1, feat_global.shape[1]))
        else:
            feat_global = feat_global.max(0)[0].expand(feat.shape[0], -1)
        feat_aug = torch.cat([feat, feat_global], 1)

        feat_aug = self.mlp_feat(feat_aug)

        feat_diff = feat - feat_aug
        feat_diff = self.mlp_diff(feat_diff)

        channel = torch.cat([feat_diff.max(1)[0].unsqueeze(1), feat_diff.mean(1).unsqueeze(1)], 1)
        atten = self.mlp_atten(channel)
        atten = torch.sigmoid(atten)

        feat_aug = feat_aug * atten

        # feat_aug = self.gcn4(feat_aug, edge_index, xyz)
        feat_aug = self.gcn5(feat_aug, edge_index)
        feat_aug = self.gcn6(feat_aug, edge_index)

        feat_aug = self.fc1(feat_aug)
        feat_aug = self.fc2(feat_aug)
        n = F.normalize(feat_aug, dim=1)
        return n


# ===========================================================================
class GATGNN(torch.nn.Module):
    def __init__(self):
        super(GATGNN, self).__init__()

        self.conv1 = GATConv(6, 32, 2)
        self.conv2 = GATConv(64, 64, 2)
        self.conv3 = GATConv(128, 128, 2)
        self.conv4 = GATConv(256, 128, 2)

        self.lin5 = GATConv(256, 64, 2)
        self.conv5 = GATConv(256, 64, 2)
        self.lin6 = GATConv(128, 32, 2)
        self.conv6 = GATConv(128, 32, 2)

        self.fc1 = nn.Linear(64, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, data_r1):

        data_r1.x = self.conv1(data_r1.x, data_r1.edge_index)
        data_r2, clust_r2 = pooling(data_r1, level=2)

        data_r2.x = self.conv2(data_r2.x, data_r2.edge_index)
        data_r3, clust_r3 = pooling(data_r2, level=2)

        data_r3.x = self.conv3(data_r3.x, data_r3.edge_index)
        data_r3.x = self.conv4(data_r3.x, data_r3.edge_index)

        feat_r2_d = data_r3.x[clust_r3]
        feat_r2_d = self.lin5(feat_r2_d, data_r2.edge_index)

        data_r2.x = torch.cat((data_r2.x, feat_r2_d), 1)
        data_r2.x = self.conv5(data_r2.x, data_r2.edge_index)

        feat_r1_d = data_r2.x[clust_r2]
        feat_r1_d = self.lin6(feat_r1_d, data_r1.edge_index)

        data_r1.x = torch.cat((data_r1.x, feat_r1_d), 1)
        feat_r1 = self.conv6(data_r1.x, data_r1.edge_index)

        feat_r1 = self.fc1(feat_r1).relu()
        feat_r1 = self.fc2(feat_r1).relu()
        feat_r1 = self.fc3(feat_r1).tanh()
        return F.normalize(feat_r1, dim=1)


class FGCNet(torch.nn.Module):
    def __init__(self):
        super(FGCNet, self).__init__()

        self.l_conv1 = FeaStConv(6, 32, 9)
        self.l_conv2 = FeaStConv(32, 64, 9)
        self.l_conv3 = FeaStConv(64, 128, 9)
        self.l_conv4 = FeaStConv(128, 128, 9)

        self.r_conv1 = FeaStConv(128, 64, 9)
        self.r_conv2 = FeaStConv(128, 64, 9)
        self.r_conv3 = FeaStConv(64, 32, 9)
        self.r_conv4 = FeaStConv(64, 32, 9)

        self.fc1 = nn.Linear(32, 1024)
        self.fc2 = nn.Linear(1024, 3)

    def forward(self, data_r1):
        data_r1.x = F.leaky_relu(self.l_conv1(data_r1.x, data_r1.edge_index), 0.1, inplace=True)
        data_r2, clust_r2 = pooling(data_r1, level=2)

        data_r2.x = F.leaky_relu(self.l_conv2(data_r2.x, data_r2.edge_index), 0.1, inplace=True)
        data_r3, clust_r3 = pooling(data_r2, level=2)

        data_r3.x = F.leaky_relu(self.l_conv3(data_r3.x, data_r3.edge_index), 0.1, inplace=True)
        data_r3.x = F.leaky_relu(self.l_conv4(data_r3.x, data_r3.edge_index), 0.1, inplace=True)

        feat_r2_r = data_r3.x[clust_r3]
        feat_r2_r = self.r_conv1(feat_r2_r, data_r2.edge_index)

        data_r2.x = torch.cat((data_r2.x, feat_r2_r), 1)
        data_r2.x = F.leaky_relu(self.r_conv2(data_r2.x, data_r2.edge_index), 0.1, inplace=True)

        feat_r1_r = data_r2.x[clust_r2]
        feat_r1_r = self.r_conv3(feat_r1_r, data_r1.edge_index)

        data_r1.x = torch.cat((data_r1.x, feat_r1_r), 1)
        feat_r1_r = F.leaky_relu(self.r_conv4(data_r1.x, data_r1.edge_index), 0.1, inplace=True)

        feat_r1_r = F.leaky_relu(self.fc1(feat_r1_r), 0.1, inplace=True)
        feat_r1_r = self.fc2(feat_r1_r)

        return F.normalize(feat_r1_r, dim=1)


class FeaStGNN_PrePool(torch.nn.Module):
    def __init__(self, in_channel=3, bias=True):
        """
        output: 'position' or 'normal'
        """
        super(FeaStGNN_PrePool, self).__init__()

        self.conv1 = FeaStConv(in_channel, 32, 6)
        self.conv2 = FeaStConv(32, 64, 6)
        self.conv3 = FeaStConv(64, 128, 6)
        self.conv4 = FeaStConv(128, 128, 6)

        self.lin5 = FeaStConv(128, 64, 6)
        self.conv5 = FeaStConv(128, 64, 6)
        self.lin6 = FeaStConv(64, 32, 6)
        self.conv6 = FeaStConv(64, 32, 6)

        self.fc1 = nn.Linear(32, 512)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, data_r1):

        data_r1.x = F.leaky_relu(self.conv1(data_r1.x, data_r1.edge_index), 0.1, inplace=True)
        data_r2 = pooling_run(data_r1, data_r1.pool_l1)

        data_r2.x = F.leaky_relu(self.conv2(data_r2.x, data_r2.edge_index), 0.1, inplace=True)
        data_r3 = pooling_run(data_r2, data_r1.pool_l2)

        data_r3.x = F.leaky_relu(self.conv3(data_r3.x, data_r3.edge_index), 0.1, inplace=True)
        data_r3.x = F.leaky_relu(self.conv4(data_r3.x, data_r3.edge_index), 0.1, inplace=True)

        feat_r2_d = data_r3.x[data_r1.pool_l2['cluster_inv']]
        feat_r2_d = self.lin5(feat_r2_d, data_r2.edge_index)
        del data_r3

        data_r2.x = torch.cat((data_r2.x, feat_r2_d), 1)
        data_r2.x = F.leaky_relu(self.conv5(data_r2.x, data_r2.edge_index), 0.1, inplace=True)
        del feat_r2_d

        feat_r1_d = data_r2.x[data_r1.pool_l1['cluster_inv']]
        feat_r1_d = self.lin6(feat_r1_d, data_r1.edge_index)
        del data_r2

        data_r1.x = torch.cat((data_r1.x, feat_r1_d), 1)
        feat_r1 = F.leaky_relu(self.conv6(data_r1.x, data_r1.edge_index), 0.1, inplace=True)
        del feat_r1_d, data_r1

        feat_r1 = F.leaky_relu(self.fc1(feat_r1), 0.1, inplace=True)
        feat_r1 = self.fc2(feat_r1)
        feat_r1 = F.normalize(feat_r1, dim=1)
        return feat_r1


# ---------------------------------------------------------------------------
class GNNModule(torch.nn.Module):
    def __init__(self, in_channel=6, pool_type='max', pool_step=2, edge_weight_type=0, wei_param=2):
        super(GNNModule, self).__init__()

        self.l_conv1 = FeaStConv(in_channel, 32, 9)
        self.pooling1 = PoolingLayer(32, pool_type, pool_step, edge_weight_type, wei_param)
        self.l_conv2 = FeaStConv(32, 64, 9)
        self.pooling2 = PoolingLayer(64, pool_type, pool_step, edge_weight_type, wei_param)
        self.l_conv3 = FeaStConv(64, 128, 9)
        self.l_conv4 = FeaStConv(128, 128, 9)

        self.r_conv1 = FeaStConv(128, 64, 9)
        self.r_conv2 = FeaStConv(128, 64, 9)
        self.r_conv3 = FeaStConv(64, 32, 9)
        self.r_conv4 = FeaStConv(64, 32, 9)

    def forward(self, data_r1, plot_pool=False):
        data_r1.x = F.leaky_relu(self.l_conv1(data_r1.x, data_r1.edge_index), 0.2, inplace=True)
        data_r2 = self.pooling1(data_r1)
        # pooled graph visualization
        if plot_pool and data_r2.pos is not None:
            nodes = data_r2.pos.detach().cpu().numpy()
            edge = data_r2.edge_index.detach().cpu().numpy().T
            data_util.plot_graph(nodes, edge)

        data_r2.x = F.leaky_relu(self.l_conv2(data_r2.x, data_r2.edge_index), 0.2, inplace=True)
        data_r3 = self.pooling2(data_r2)
        if plot_pool and data_r3.pos is not None:
            nodes = data_r3.pos.detach().cpu().numpy()
            edge = data_r3.edge_index.detach().cpu().numpy().T
            data_util.plot_graph(nodes, edge)

        data_r3.x = F.leaky_relu(self.l_conv3(data_r3.x, data_r3.edge_index), 0.2, inplace=True)
        data_r3.x = F.leaky_relu(self.l_conv4(data_r3.x, data_r3.edge_index), 0.2, inplace=True)

        feat_r2_r = self.pooling2.unpooling(data_r3.x)
        feat_r2_r = self.r_conv1(feat_r2_r, data_r2.edge_index)

        data_r2.x = torch.cat((data_r2.x, feat_r2_r), 1)
        data_r2.x = F.leaky_relu(self.r_conv2(data_r2.x, data_r2.edge_index), 0.2, inplace=True)

        feat_r1_r = self.pooling1.unpooling(data_r2.x)
        feat_r1_r = self.r_conv3(feat_r1_r, data_r1.edge_index)

        data_r1.x = torch.cat((data_r1.x, feat_r1_r), 1)
        feat_r1_r = F.leaky_relu(self.r_conv4(data_r1.x, data_r1.edge_index), 0.2, inplace=True)
        return feat_r1_r


class DualGNN(torch.nn.Module):
    def __init__(self, force_depth=False, pool_type='max', edge_weight_type=0, wei_param=2):
        super(DualGNN, self).__init__()
        self.force_depth = force_depth

        # graph-v
        self.gnn_v = GNNModule(6, pool_type, 2, edge_weight_type, wei_param)  # out_channel = 32
        self.fc_v1 = nn.Linear(32, 1024)
        self.fc_v2 = nn.Linear(1024, 1) if self.force_depth else nn.Linear(1024, 3)

        # graph-f
        self.gnn_f = GNNModule(12, pool_type, 2, edge_weight_type, wei_param)  # out_channel = 32
        self.fc_f1 = nn.Linear(32, 1024)
        self.fc_f2 = nn.Linear(1024, 3)

    def forward(self, dual_data):
        data_v, data_f = dual_data
        xyz = data_v.x[:, :3]
        nf = data_f.x[:, 3:6]

        feat_v = self.gnn_v(data_v)
        feat_v_h = F.leaky_relu(self.fc_v1(feat_v), 0.2, inplace=True)
        feat_v = self.fc_v2(feat_v_h)
        if self.force_depth:  # constrain coordinate in the depth direction of original coordinate frame
            feat_v = feat_v * data_v.depth_direction

            # feat_v = (feat_v*data_v.depth_direction).sum(1, keepdim=True)
            # feat_v = feat_v * data_v.depth_direction

        feat_v += xyz

        # new node feature of facet graph
        face_cent = feat_v[data_f.fv_indices].mean(1)
        face_norm = data_util.computer_face_normal(feat_v, data_f.fv_indices)
        data_f.x = torch.cat((data_f.x, face_cent, face_norm), 1)

        feat_f = self.gnn_f(data_f)
        feat_f_h = F.leaky_relu(self.fc_f1(feat_f), 0.2, inplace=True)
        feat_f = self.fc_f2(feat_f_h)

        return feat_v, F.normalize(feat_f, dim=1), None


class SingleGNN(torch.nn.Module):
    def __init__(self, out_type, force_depth=False, pool_type='max', pool_step=2, edge_weight_type=0):
        super(SingleGNN, self).__init__()
        assert (out_type in ['normal', 'vertex'])
        self.out_type = out_type
        self.force_depth = force_depth

        self.gnn = GNNModule(6, pool_type, pool_step, edge_weight_type)  # out_channel = 32
        self.fc_v1 = nn.Linear(32, 1024)
        self.fc_v2 = nn.Linear(1024, 1) if self.force_depth else nn.Linear(1024, 3)

        self.fc_mat = nn.Linear(1024, 4)

    def forward(self, data):
        if self.out_type == 'vertex':
            x_init = data.x[:, :3]
        elif self.out_type == 'normal':
            x_init = data.x[:, 3:6]

        feat = self.gnn(data)
        feat = F.leaky_relu(self.fc_v1(feat), 0.2, inplace=True)

        quat = feat.max(0)[0].unsqueeze(0)
        quat = self.fc_mat(quat)
        quat = quat + quat.new_tensor([1, 0, 0, 0])
        quat = F.normalize(quat, dim=1)
        mat = batch_quat_to_rotmat(quat, normalize=False)

        feat = self.fc_v2(feat)
        if self.force_depth:  # constrain coordinate in the depth direction of original coordinate frame
            feat = feat * data.depth_direction
        feat += x_init

        if self.out_type == 'vertex':
            feat = torch.bmm(feat.unsqueeze(0), mat).squeeze()
            return feat
        elif self.out_type == 'normal':
            return F.normalize(feat, dim=1)


# ---------------------------------------------------------------------------
class DualLinear(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DualLinear, self).__init__()

        self.lin_v = nn.Linear(in_channel, out_channel)
        self.lin_f = nn.Linear(in_channel, out_channel)

    def forward(self, data_v, data_f):
        feat_v = F.leaky_relu(self.lin_v(data_v.x), 0.2, inplace=True)
        feat_f = F.leaky_relu(self.lin_f(data_f.x), 0.2, inplace=True)
        return feat_v, feat_f


class DualConv(torch.nn.Module):
    def __init__(self, in_channel, out_channel, head):
        super(DualConv, self).__init__()

        self.conv_v = FeaStConv(in_channel, out_channel, head)
        self.conv_f = FeaStConv(in_channel, out_channel, head)

    def forward(self, data_v, data_f):
        data_v.x = F.leaky_relu(self.conv_v(data_v.x, data_v.edge_index), 0.2, inplace=True)
        data_f.x = F.leaky_relu(self.conv_f(data_f.x, data_f.edge_index), 0.2, inplace=True)
        return data_v, data_f


class DualPooling(torch.nn.Module):
    def __init__(self, in_channel, pool_type='max', pool_step=2, edge_weight_type=0):
        super(DualPooling, self).__init__()

        self.pool_v = PoolingLayer(in_channel, pool_type, pool_step, edge_weight_type)
        self.pool_f = PoolingLayer(in_channel, pool_type, pool_step, edge_weight_type)

    def forward(self, data_v, data_f):
        data_v = self.pool_v(data_v)
        data_f = self.pool_f(data_f)
        return data_v, data_f

    def unpooling(self, data_v, data_f):
        feat_v = self.pool_v.unpooling(data_v.x)
        feat_f = self.pool_f.unpooling(data_f.x)
        return feat_v, feat_f


class DualGNN_Fusion(torch.nn.Module):
    def __init__(self, force_depth=False, pool_type='max', pool_step=2, edge_weight_type=0):
        super(DualGNN_Fusion, self).__init__()
        self.force_depth = force_depth

        self.lin_in = DualLinear(6, 32)

        self.dual_conv_l1 = DualConv(32, 32, 9)
        self.dual_pool1 = DualPooling(32, pool_type, pool_step, edge_weight_type)
        self.dual_conv_l2 = DualConv(32, 64, 9)
        self.dual_pool2 = DualPooling(64, pool_type, pool_step, edge_weight_type)
        self.dual_conv_l3 = DualConv(64, 128, 9)

        self.dual_conv_r3 = DualConv(128, 64, 9)
        self.dual_conv_r2 = DualConv(128, 32, 9)
        self.dual_conv_r1 = DualConv(64, 32, 9)

        self.fusion3 = DualFusionLayer(64)
        self.fusion2 = DualFusionLayer(32)
        self.fusion1 = DualFusionLayer(32)

        self.dual_conv_out = DualConv(32, 32, 9)
        self.lin_out1 = DualLinear(32, 512)
        self.lin_out_v = nn.Linear(512, 3)
        self.lin_out_f = nn.Linear(512, 3)

    def forward(self, dual_data):
        data_v_r1, data_f_r1 = dual_data
        xyz = data_v_r1.x[:, :3]
        nf = data_f_r1.x[:, 3:6]

        data_v_r1.x, data_f_r1.x = self.lin_in(data_v_r1, data_f_r1)

        data_v_r1, data_f_r1 = self.dual_conv_l1(data_v_r1, data_f_r1)
        data_v_r2, data_f_r2 = self.dual_pool1(data_v_r1, data_f_r1)
        data_v_r2, data_f_r2 = self.dual_conv_l2(data_v_r2, data_f_r2)
        data_v_r3, data_f_r3 = self.dual_pool2(data_v_r2, data_f_r2)
        data_v_r3, data_f_r3 = self.dual_conv_l3(data_v_r3, data_f_r3)

        data_v_r3, data_f_r3 = self.dual_conv_r3(data_v_r3, data_f_r3)
        data_v_r3.x, data_f_r3.x = self.fusion3(data_v_r3, data_f_r3)

        feat_v_r2, feat_f_r2 = self.dual_pool2.unpooling(data_v_r3, data_f_r3)
        data_v_r2.x = torch.cat((data_v_r2.x, feat_v_r2), 1)
        data_f_r2.x = torch.cat((data_f_r2.x, feat_f_r2), 1)
        data_v_r2, data_f_r2 = self.dual_conv_r2(data_v_r2, data_f_r2)
        data_v_r2.x, data_f_r2.x = self.fusion2(data_v_r2, data_f_r2)

        feat_v_r1, feat_f_r1 = self.dual_pool1.unpooling(data_v_r2, data_f_r2)
        data_v_r1.x = torch.cat((data_v_r1.x, feat_v_r1), 1)
        data_f_r1.x = torch.cat((data_f_r1.x, feat_f_r1), 1)
        data_v_r1, data_f_r1 = self.dual_conv_r1(data_v_r1, data_f_r1)
        data_v_r1.x, data_f_r1.x = self.fusion1(data_v_r1, data_f_r1)

        data_v_r1, data_f_r1 = self.dual_conv_out(data_v_r1, data_f_r1)
        feat_v, feat_f = self.lin_out1(data_v_r1, data_f_r1)
        feat_v = self.lin_out_v(feat_v)
        feat_f = self.lin_out_f(feat_f)

        feat_v += xyz
        feat_f += nf
        return feat_v, F.normalize(feat_f, dim=1), None


class DualGNN_Fusion_temp(torch.nn.Module):
    def __init__(self, force_depth=False, pool_type='max', pool_step=2, edge_weight_type=0):
        super(DualGNN_Fusion_temp, self).__init__()
        self.force_depth = force_depth

        self.lin_in = DualLinear(6, 32)

        self.dual_conv_l1 = DualConv(32, 32, 9)
        self.dual_pool1 = DualPooling(32, pool_type, pool_step, edge_weight_type)
        self.dual_conv_l2 = DualConv(32, 64, 9)
        self.dual_pool2 = DualPooling(64, pool_type, pool_step, edge_weight_type)
        self.dual_conv_l3 = DualConv(64, 128, 9)

        self.fusion_l1 = DualFusionLayer(32)
        self.fusion_l2 = DualFusionLayer(64)
        self.fusion_l3 = DualFusionLayer(128)

        self.dual_conv_r3 = DualConv(128, 64, 9)
        self.dual_conv_r2 = DualConv(128, 32, 9)
        self.dual_conv_r1 = DualConv(64, 32, 9)

        self.fusion_r3 = DualFusionLayer(64)
        self.fusion_r2 = DualFusionLayer(32)
        self.fusion_r1 = DualFusionLayer(32)

        self.dual_conv_out = DualConv(32, 32, 9)
        self.lin_out1 = DualLinear(32, 512)
        self.lin_out_v = nn.Linear(512, 3)
        self.lin_out_f = nn.Linear(512, 3)

    def forward(self, dual_data):
        data_v_r1, data_f_r1 = dual_data
        xyz = data_v_r1.x[:, :3]
        nf = data_f_r1.x[:, 3:6]

        data_v_r1.x, data_f_r1.x = self.lin_in(data_v_r1, data_f_r1)

        data_v_r1, data_f_r1 = self.dual_conv_l1(data_v_r1, data_f_r1)
        data_v_r1.x, data_f_r1.x = self.fusion_l1(data_v_r1, data_f_r1)

        data_v_r2, data_f_r2 = self.dual_pool1(data_v_r1, data_f_r1)
        data_v_r2, data_f_r2 = self.dual_conv_l2(data_v_r2, data_f_r2)
        data_v_r2.x, data_f_r2.x = self.fusion_l2(data_v_r2, data_f_r2)

        data_v_r3, data_f_r3 = self.dual_pool2(data_v_r2, data_f_r2)
        data_v_r3, data_f_r3 = self.dual_conv_l3(data_v_r3, data_f_r3)
        data_v_r3.x, data_f_r3.x = self.fusion_l3(data_v_r3, data_f_r3)

        data_v_r3, data_f_r3 = self.dual_conv_r3(data_v_r3, data_f_r3)
        data_v_r3.x, data_f_r3.x = self.fusion_r3(data_v_r3, data_f_r3)

        feat_v_r2, feat_f_r2 = self.dual_pool2.unpooling(data_v_r3, data_f_r3)
        data_v_r2.x = torch.cat((data_v_r2.x, feat_v_r2), 1)
        data_f_r2.x = torch.cat((data_f_r2.x, feat_f_r2), 1)
        data_v_r2, data_f_r2 = self.dual_conv_r2(data_v_r2, data_f_r2)
        data_v_r2.x, data_f_r2.x = self.fusion_r2(data_v_r2, data_f_r2)

        feat_v_r1, feat_f_r1 = self.dual_pool1.unpooling(data_v_r2, data_f_r2)
        data_v_r1.x = torch.cat((data_v_r1.x, feat_v_r1), 1)
        data_f_r1.x = torch.cat((data_f_r1.x, feat_f_r1), 1)
        data_v_r1, data_f_r1 = self.dual_conv_r1(data_v_r1, data_f_r1)
        data_v_r1.x, data_f_r1.x = self.fusion_r1(data_v_r1, data_f_r1)

        data_v_r1, data_f_r1 = self.dual_conv_out(data_v_r1, data_f_r1)
        feat_v, feat_f = self.lin_out1(data_v_r1, data_f_r1)
        feat_v = self.lin_out_v(feat_v)
        feat_f = self.lin_out_f(feat_f)

        feat_v += xyz
        feat_f += nf
        return feat_v, F.normalize(feat_f, dim=1), None


def _laplacian(v, edge_idx_v, normal=None):
    row, col = edge_idx_v
    edge = v[row] - v[col]
    lap = scatter(edge, row, dim=0, reduce='mean')
    if normal is not None:
        lap = normal * (lap*normal).sum(1, keepdim=True)
    return lap


def laplacian_loss(vp, v, edge_idx_v, normal=None):
    edge_idx_v, _ = remove_self_loops(edge_idx_v)
    lap_vp = _laplacian(vp, edge_idx_v, normal)
    lap_v = _laplacian(v, edge_idx_v, normal)
    loss = (lap_vp-lap_v).abs().sum(1).mean()
    return loss


def loss_v(vp, v, dis='L2', apply_icp=False):
    if apply_icp:
        icp_rst = icp(vp.unsqueeze(0), v.unsqueeze(0))
        vp = icp_rst.Xt.squeeze()

    if dis == 'CD':
        loss = chamfer_distance(vp.unsqueeze(0), v.unsqueeze(0)).squeeze()
    elif dis == 'EMD':
        pass
    elif dis == 'L1':
        loss = (vp-v).abs().sum(1).mean()
    elif dis == 'L2':
        loss = (vp-v).pow(2).sum(1).mean()
    return loss


def loss_n(np, n, norm='L1', fc_p=None, fc=None):
    if norm == 'L1':
        loss = (np-n).abs().sum(1).mean()
    elif norm == 'L2':
        loss = (np-n).pow(2).sum(1).mean()
    elif norm == 'sided':
        dis, idx = sided_distance(fc_p.unsqueeze(0), fc.unsqueeze(0))
        idx = idx.squeeze()
        loss = (np-n[idx]).abs().sum(1).mean()
    return loss


def dual_loss(loss_v, loss_n, v_scale=1, n_scale=1, alpha=None):
    if alpha is None:
        return loss_v*v_scale + loss_n*n_scale
    else:
        return alpha*loss_v*v_scale + (1-alpha)*loss_n*n_scale


def error_v(vp, v):
    """
    Euclidean distance
    """
    error = (vp-v).pow(2).sum(1).pow(0.5)
    return error.mean()


def error_n(np, n):
    """
    Intersection angular
    """
    error = (np-n).pow(2).sum(1)
    val = torch.clamp(1-error/2, min=-1, max=1)
    return (torch.acos(val) * 180 / math.pi).mean()


# main function to save net model for visualization
if __name__ == "__main__":
    from dataset import DualDataset

    data_file = R"E:\code\python\denoising\TempNet\data\Synthetic\train\processed_data\Cylinder_n2.pt"
    dual_data = torch.load(data_file)
    dual_data = DualDataset.post_processing(dual_data, 'Synthetic')

    net = DualGNN_Fusion()
    y = net(dual_data)



    total_params = sum(p.numel() for p in conv.parameters())
    print(F"{total_params:} total parameters")
    x = torch.Tensor(1, 1, 3, 3)
    y = conv(x)

    # =====================================================
    model = GGNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(F"{total_params:} total parameters")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(F"{total_trainable_params:,} training parameters")

    import torchvision
    data = torch.randn(64, 3, 324, 324)
    model = torchvision.models.resnet18()
    model_name = os.path.join(BASE_DIR, "resnet18_visual.onnx")

    torch.onnx.export(model, data, model_name, export_params=True)
    # https://lutzroeder.github.io/netron/

    print(F'"{model_name}"  saved.')
