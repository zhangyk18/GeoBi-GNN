import os
import sys
import glob
import argparse
from datetime import datetime
import numpy as np
import openmesh as om
import torch
import data_util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'log')


def is_debug():
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    elif gettrace():
        return True
    else:
        return False


# ================================================== train =====================================================
def predict_one_submesh(net, device, data):
    with torch.no_grad():
        data = data.to(device)
        return net(data)


def normalize_data(data_v, data_f, data_type):
    # vertex graph
    v_pos_normalized = (data_v.pos - data_v.centroid) * data_v.scale
    data_v.x = torch.cat((v_pos_normalized, data_v.normal), 1)
    data_v.normal = None

    # facet graph
    f_pos_normalized = (data_f.pos - data_v.centroid) * data_v.scale
    data_f.x = torch.cat((f_pos_normalized, data_f.normal), 1)
    data_f.pos = data_f.normal = None

    if data_type not in ['Kinect_v1', 'Kinect_v2']:
        data_v.pos = data_v.depth_direction = None
    return data_v, data_f


def predict_one_bak(opt, net, device, filename, rst_filename=None, filename_gt=None):
    from dataset import DualDataset
    from network import error_n

    time_start = datetime.now()
    # 1.load data
    mesh_noisy = om.read_trimesh(filename)
    points_noisy = mesh_noisy.points().astype(np.float32)

    # 2.center and scal
    _, centroid, scale = data_util.center_and_scale(points_noisy, mesh_noisy.ev_indices(), s_type=0)

    # 3.split to submeshes and inference
    if mesh_noisy.n_faces() <= opt.sub_size:
        data = DualDataset.process_one_submesh(mesh_noisy)
        data[0].centroid = torch.from_numpy(centroid).float()
        data[0].scale = scale
        data = normalize_data(data[0], data[1], opt.data_type)
        data = data[0] if opt.out_type == 'vertex' else data[1]
        Yp = predict_one_submesh(net, device, data)
    else:
        if opt.out_type == 'vertex':
            Yp = torch.zeros((mesh_noisy.n_vertices(), 3), dtype=torch.float32, device=device)
            sum_v = torch.zeros((mesh_noisy.n_vertices(), 1), dtype=torch.int8, device=device)
        else:
            Yp = torch.zeros((mesh_noisy.n_faces(), 3), dtype=torch.float32, device=device)

        flag_f = np.zeros(mesh_noisy.n_faces(), dtype=np.bool)
        fv_indices = mesh_noisy.fv_indices()
        vf_indices = mesh_noisy.vf_indices()
        face_cent = points_noisy[fv_indices].mean(1)

        seed = np.argmax(((face_cent - centroid)**2).sum(1))
        for sub_num in range(1, sys.maxsize):
            # select patch facet indices
            select_faces = data_util.mesh_get_neighbor_np(fv_indices, vf_indices, seed, neighbor_count=opt.sub_size)
            flag_f.put(select_faces, True)

            # split submesh based on facet indices
            V_idx, F = data_util.get_submesh(fv_indices, select_faces)
            sum_v[V_idx] += 1

            # to Data
            submesh_n = om.TriMesh(points_noisy[V_idx], F)
            data = DualDataset.process_one_submesh(submesh_n)
            data[0].centroid = torch.from_numpy(centroid).float()
            data[0].scale = scale
            data = normalize_data(data[0], data[1], opt.data_type)
            data = data[0] if opt.out_type == 'vertex' else data[1]
            Y_sub = predict_one_submesh(net, device, data)
            if opt.out_type == 'vertex':
                Yp[V_idx] += Y_sub
            else:
                Yp[select_faces] += Y_sub

            # whether all facets have been visited, next seed
            left_idx = np.where(~flag_f)[0]
            if left_idx.size:
                idx_temp = np.argmax(((face_cent[left_idx] - points_noisy.mean(0))**2).sum(1))
                seed = left_idx[idx_temp]
            else:
                break
        if opt.out_type == 'vertex':
            Yp /= sum_v
        else:
            Yp = torch.nn.functional.normalize(Yp, dim=1)

    # 4.update position and save
    if opt.out_type == 'vertex':
        Yp = Yp.cpu() / scale + centroid
        fv_indices = torch.from_numpy(mesh_noisy.fv_indices()).long()
        om.write_mesh(rst_filename, om.TriMesh(Yp.numpy(), mesh_noisy.fv_indices()))
        Np = data_util.computer_face_normal(Yp, fv_indices)
        V_new = Yp
    else:
        Yp = Yp.cpu()
        Vn = torch.from_numpy(points_noisy).float()
        fv_indices = torch.from_numpy(mesh_noisy.fv_indices()).long()
        vf_indices = torch.from_numpy(mesh_noisy.vf_indices()).long()
        V_new = data_util.update_position(Vn, fv_indices, vf_indices, Yp, 20, depth_direction=torch.nn.functional.normalize(Vn, dim=1) if opt.force_depth else None)
        om.write_mesh(rst_filename, om.TriMesh(V_new.numpy(), mesh_noisy.fv_indices()))
        Np = Yp

    angle1 = angle2 = 0
    if filename_gt is not None:
        mesh_o = om.read_trimesh(filename_gt)
        mesh_o.update_face_normals()
        Nt = torch.from_numpy(mesh_o.face_normals()).float()
        angle1 = error_n(Np, Nt)
        Np2 = data_util.computer_face_normal(V_new, fv_indices)
        angle2 = error_n(Np2, Nt)

    print(F"angle1: {angle1:9.6f},  angle2: {angle2:9.6f},  faces: {Np.shape[0]:>6},  time: {(datetime.now()-time_start).total_seconds():7.4f} s,  '{os.path.basename(rst_filename)}'")
    return angle1, angle2, Np.shape[0]


def predict_one(opt, net, device, filename, rst_filename=None, filename_gt=None):
    from dataset import DualDataset
    from network import error_n

    time_start = datetime.now()
    # 1.load data
    mesh_noisy = om.read_trimesh(filename)
    points_noisy = mesh_noisy.points().astype(np.float32)

    # 2.center and scal
    _, centroid, scale = data_util.center_and_scale(points_noisy, mesh_noisy.ev_indices(), s_type=0)

    # 3.split to submeshes and inference
    if mesh_noisy.n_faces() <= opt.sub_size:
        data = DualDataset.process_one_submesh(mesh_noisy)
        # data[0].centroid = torch.from_numpy(centroid).float()
        # data[0].scale = scale
        data = normalize_data(data[0], data[1], opt.data_type)
        data = data[0] if opt.out_type == 'vertex' else data[1]
        Yp = predict_one_submesh(net, device, data)

        Yp = Yp / data.scale + data.centroid
    else:
        Yp = torch.zeros((mesh_noisy.n_vertices(), 3), dtype=torch.float32, device=device)
        sum_v = torch.zeros((mesh_noisy.n_vertices(), 1), dtype=torch.int8, device=device)

        flag_f = np.zeros(mesh_noisy.n_faces(), dtype=np.bool)
        fv_indices = mesh_noisy.fv_indices()
        vf_indices = mesh_noisy.vf_indices()
        face_cent = points_noisy[fv_indices].mean(1)

        seed = np.argmax(((face_cent - centroid)**2).sum(1))
        for sub_num in range(1, sys.maxsize):
            # select patch facet indices
            select_faces = data_util.mesh_get_neighbor_np(fv_indices, vf_indices, seed, neighbor_count=opt.sub_size)
            flag_f.put(select_faces, True)

            # split submesh based on facet indices
            V_idx, F = data_util.get_submesh(fv_indices, select_faces)
            sum_v[V_idx] += 1

            # to Data
            submesh_n = om.TriMesh(points_noisy[V_idx], F)
            data = DualDataset.process_one_submesh(submesh_n)
            # data[0].centroid = torch.from_numpy(centroid).float()
            # data[0].scale = scale
            data = normalize_data(data[0], data[1], opt.data_type)
            data = data[0] if opt.out_type == 'vertex' else data[1]
            Y_sub = predict_one_submesh(net, device, data)
            Y_sub = Y_sub / data.scale + data.centroid
            Yp[V_idx] += Y_sub

            # whether all facets have been visited, next seed
            left_idx = np.where(~flag_f)[0]
            if left_idx.size:
                idx_temp = np.argmax(((face_cent[left_idx] - points_noisy.mean(0))**2).sum(1))
                seed = left_idx[idx_temp]
            else:
                break
        Yp /= sum_v

    Yp = Yp.cpu()
    # 4.update position and save
    fv_indices = torch.from_numpy(mesh_noisy.fv_indices()).long()
    om.write_mesh(rst_filename, om.TriMesh(Yp.numpy(), mesh_noisy.fv_indices()))
    Np = data_util.computer_face_normal(Yp, fv_indices)
    V_new = Yp

    angle1 = angle2 = 0
    if filename_gt is not None:
        mesh_o = om.read_trimesh(filename_gt)
        mesh_o.update_face_normals()
        Nt = torch.from_numpy(mesh_o.face_normals()).float()
        angle1 = error_n(Np, Nt)
        Np2 = data_util.computer_face_normal(V_new, fv_indices)
        angle2 = error_n(Np2, Nt)

    print(F"angle1: {angle1:9.6f},  angle2: {angle2:9.6f},  faces: {Np.shape[0]:>6},  time: {(datetime.now()-time_start).total_seconds():7.4f} s,  '{os.path.basename(rst_filename)}'")
    return angle1, angle2, Np.shape[0]


def predict_dir(params_path, data_dir=None, sub_size=None, gpu=-1):
    assert (data_dir is None or os.path.exists(data_dir))

    opt = torch.load(params_path)
    opt.sub_size = opt.sub_size if sub_size is None else sub_size
    print('\n' + str(opt) + '\n')
    bak_dir = os.path.dirname(params_path)

    # 1. prepare data
    filenames = []
    filenames_gt = []
    if data_dir is None:
        data_dir = os.path.join(DATA_DIR, opt.data_type)
        original_dir = os.path.join(DATA_DIR, opt.data_type, 'original')
        data_list = list(filter(None, [line.strip() for line in open(os.path.join(data_dir, "test_list.txt"))]))
        for name in data_list:
            files_n = glob.glob(os.path.join(data_dir, 'noisy', F"{name}-n*.obj"))
            for name_n in files_n:
                filenames.append(name_n)
                filenames_gt.append(os.path.join(original_dir, F"{name}.obj"))
    else:
        filenames = glob.glob(os.path.join(data_dir, '*.obj'))

    print(F"\nInfer {opt.flag}, sub_size:{opt.sub_size}, {len(filenames)} files ...\n")
    result_dir = os.path.join(data_dir, F"result_{opt.flag}")
    os.makedirs(result_dir, exist_ok=True)

    # model
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    elif gpu >= 0:
        device = torch.device(F"cuda:{gpu}")
    else:
        device = torch.device("cuda")

    sys.path.insert(0, bak_dir)
    import network
    net = network.SingleGNN(out_type=opt.out_type, force_depth=opt.force_depth, pool_type=opt.pool_type, pool_step=2, edge_weight_type=opt.wei_type)
    net.load_state_dict(torch.load(os.path.join(bak_dir, opt.model_name)))
    net = net.to(device)
    net.eval()

    # inference
    error_all = np.zeros((3, len(filenames)))  # count, mean_error
    for i, noisy_file in enumerate(filenames):
        rst_file = os.path.join(result_dir, os.path.basename(noisy_file))
        if os.path.exists(rst_file):
            continue
        # angle1, angle2, count = predict_one(opt, net, device, noisy_file, rst_file, None if len(filenames_gt) == 0 else filenames_gt[i])
        angle1, angle2, count = predict_one_bak(opt, net, device, noisy_file, rst_file, None if len(filenames_gt) == 0 else filenames_gt[i])
        error_all[0, i] = count
        error_all[1, i] = angle1
        error_all[2, i] = angle2

    count_sum = error_all[0].sum().astype(np.int32)
    error_mean1 = (error_all[0]*error_all[1]).sum() / count_sum
    error_mean2 = (error_all[0]*error_all[2]).sum() / count_sum
    print(F"\nNum_face: {count_sum:>6},  angle_mean1: {error_mean1:.6f},  angle_mean1: {error_mean2:.6f}")

    print("\n--- end ---")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--params_path', type=str, default=None, help='params_path')
    parser.add_argument('--data_dir', type=str, default=None, help='data_dir')
    parser.add_argument('--sub_size', type=int, default=None, help='submesh size')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu index')
    opt = parser.parse_args()

    if is_debug():
        opt.params_path = R"E:\code\python\denoising\TempNet\log\Single-GNN_Kinect_Fusion_data_aug\20210228-125814\Single-GNN_Kinect_Fusion_params.pth"
        # opt.data_dir = R"E:\SysFile\Desktop\comparison_CNF\synthetic"
        opt.sub_size = 100000
        opt.gpu = -1

    predict_dir(opt.params_path, data_dir=opt.data_dir, sub_size=opt.sub_size, gpu=opt.gpu)

    # python test_single.py --params_path= --gpu=

    # download data from server via SSH
    # scp -r -P 25194 zyk@10.10.1.150:[source_path] [local_path]
