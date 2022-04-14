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
def predict_one_submesh(net, device, dual_data):
    with torch.no_grad():
        dual_data = [d.to(device) for d in dual_data]
        vert_p, norm_p, alpha = net(dual_data)
        return vert_p, norm_p


def predict_one(opt, net, device, filename, rst_filename=None, filename_gt=None):
    from dataset import DualDataset
    from network import error_n

    time_start = datetime.now()
    # 1.load data
    mesh_noisy = om.read_trimesh(filename)
    points_noisy = mesh_noisy.points().astype(np.float32)

    # # for debug
    # if mesh_noisy.n_faces() > 30000:
    #     return

    # 2.center and scal
    _, centroid, scale = data_util.center_and_scale(points_noisy, mesh_noisy.edge_vertex_indices())

    # 3.split to submeshes and inference
    if mesh_noisy.n_faces() <= opt.sub_size:
        dual_data = DualDataset.process_one_submesh(mesh_noisy)
        dual_data[0].centroid = torch.from_numpy(centroid).float()
        dual_data[0].scale = scale
        dual_data = DualDataset.post_processing(dual_data, opt.data_type, True)
        Vp, Np = predict_one_submesh(net, device, dual_data)
    else:
        # mesh_noisy.update_face_normals()
        # Nn = mesh_noisy.face_normals().astype(np.float32)

        Vp = torch.zeros((mesh_noisy.n_vertices(), 3), dtype=torch.float32, device=device)
        Np = torch.zeros((mesh_noisy.n_faces(), 3), dtype=torch.float32, device=device)
        sum_v = torch.zeros((Vp.shape[0], 1), dtype=torch.int8, device=device)

        flag_f = np.zeros(Np.shape[0], dtype=np.bool)
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
            dual_data = DualDataset.process_one_submesh(submesh_n)
            dual_data[0].centroid = torch.from_numpy(centroid).float()
            dual_data[0].scale = scale
            dual_data = DualDataset.post_processing(dual_data, opt.data_type, True)
            vert_p, norm_p = predict_one_submesh(net, device, dual_data)
            Vp[V_idx] += vert_p
            Np[select_faces] += norm_p

            # whether all facets have been visited, next seed
            left_idx = np.where(~flag_f)[0]
            if left_idx.size:
                idx_temp = np.argmax(((face_cent[left_idx] - centroid)**2).sum(1))
                seed = left_idx[idx_temp]
            else:
                break
        Vp /= sum_v
        Np = torch.nn.functional.normalize(Np, dim=1)

    # 4.update position and save
    Vp = Vp.cpu() / scale + centroid
    if mesh_noisy.n_faces() > opt.sub_size:
        nan_idx = (sum_v == 0).squeeze()
        Vp[nan_idx] = torch.from_numpy(points_noisy)[nan_idx]
    Np = Np.cpu()
    om.write_mesh(F"{rst_filename[:-4]}-v.obj", om.TriMesh(Vp.cpu().numpy(), mesh_noisy.fv_indices()))

    fv_indices = torch.from_numpy(mesh_noisy.fv_indices()).long()
    vf_indices = torch.from_numpy(mesh_noisy.vf_indices()).long()
    depth_direction = None
    if opt.data_type in ['Kinect_v1', 'Kinect_v2']:
        depth_direction = torch.nn.functional.normalize(torch.from_numpy(points_noisy), dim=1)
        # dd = torch.nn.functional.normalize(Vp, dim=1)
    # n_iter=60 for Kinect_Fusion, refer to <<Mesh Denoising with Facet Graph Convolutions>>
    V1 = data_util.update_position2(Vp, fv_indices, vf_indices, Np, 60, depth_direction=depth_direction)
    # V2 = data_util.update_position(Vp, fv_indices, vf_indices, Np, 60, depth_direction=depth_direction)
    # aa = torch.abs(V1-V2).max()
    # print(F"{aa:.4f}")

    # om.write_mesh(F"{rst_filename[:-4]}-v.obj", om.TriMesh(Vp.numpy(), mesh_noisy.fv_indices()))
    om.write_mesh(rst_filename, om.TriMesh(V1.numpy(), mesh_noisy.fv_indices()))

    angle1 = angle2 = 0
    if filename_gt is not None:
        mesh_o = om.read_trimesh(filename_gt)
        mesh_o.update_face_normals()
        Nt = torch.from_numpy(mesh_o.face_normals()).float()
        angle1 = error_n(Np, Nt)
        Np2 = data_util.computer_face_normal(V, fv_indices)
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
        data_dir = os.path.join(DATA_DIR, opt.data_type, 'test')
        original_dir = os.path.join(DATA_DIR, opt.data_type, 'test', 'original')
        data_list = glob.glob(os.path.join(original_dir, "*.obj"))
        data_list = [os.path.basename(d)[:-4] for d in data_list]
        for name in data_list:
            files_n = glob.glob(os.path.join(data_dir, 'noisy', F"{name}_n*.obj"))
            for name_n in files_n:
                filenames.append(name_n)
                filenames_gt.append(os.path.join(original_dir, F"{name}.obj"))
    else:
        filenames = glob.glob(os.path.join(data_dir, '*.obj'))

    print(F"\nInfer {opt.flag}, sub_size:{opt.sub_size}, {len(filenames)} files ...\n")
    result_dir = os.path.join(data_dir, F"result_{opt.flag}")
    os.makedirs(result_dir, exist_ok=True)

    # 2. model
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    elif gpu >= 0:
        device = torch.device(F"cuda:{gpu}")
    else:
        device = torch.device("cuda")
    # device = torch.device("cpu")

    sys.path.insert(0, bak_dir)
    import network
    net = network.DualGNN(
        force_depth=opt.force_depth, include_facet_initial_feature=opt.include_facet_initial_feature,
        pool_type=opt.pool_type, pool_step=2, edge_weight_type=opt.wei_type, learn_res=opt.learn_res)
    # net = network.DualGNN_Fusion(force_depth=opt.force_depth, pool_type=opt.pool_type, pool_step=2, edge_weight_type=opt.wei_type)
    net.load_state_dict(torch.load(os.path.join(bak_dir, opt.model_name)))
    net = net.to(device)
    net.eval()

    # 3. infer
    error_all = np.zeros((3, len(filenames)))  # count, mean_error
    for i, noisy_file in enumerate(filenames):
        rst_file = os.path.join(result_dir, os.path.basename(noisy_file))
        # if os.path.exists(rst_file):
        #     continue
        if 'joint_n3' not in noisy_file:
            continue
        angle1, angle2, count = predict_one(opt, net, device, noisy_file, rst_file, None if len(filenames_gt) == 0 else filenames_gt[i])
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
        opt.params_path = R"E:\code\python\denoising\TempNet\log\Bi-GNN_Synthetic_abla\20210309-145738\Bi-GNN_Synthetic_params.pth"
        # opt.data_dir = R"E:\data\dataset\Biwi_Kinect_Head_Pose-Database\02\mesh"
        opt.sub_size = 100000
        opt.gpu = 1

    predict_dir(opt.params_path, data_dir=opt.data_dir, sub_size=opt.sub_size, gpu=opt.gpu)

    # python test_dual.py --params_path=  --original_dir=  --gpu=  --sub_size=

    # download data from server via SSH
    # scp -r -P 25180 zyk@10.10.1.150:[source_path] [local_path]
