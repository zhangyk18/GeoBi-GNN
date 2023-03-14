# -*- coding: utf-8 -*-
import os, sys, glob, argparse
from datetime import datetime
import numpy as np
import openmesh as om
import torch
import data_util


CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
LOG_DIR = os.path.join(BASE_DIR, 'log')
IS_DEBUG = getattr(sys, 'gettrace', None) is not None and sys.gettrace()


# ================================================== train =====================================================
def predict_one_submesh(net, device, dual_data):
    with torch.no_grad():
        dual_data = [d.to(device) for d in dual_data]
        vert_p, norm_p, alpha = net(dual_data)
        return vert_p, norm_p


def predict_one(opt, net, device, filename, rst_filename=None, filename_gt=None):
    from dataset import DualDataset
    from network import error_n

    # 1.load data
    mesh_noisy = om.read_trimesh(filename)
    points_noisy = mesh_noisy.points().astype(np.float32)


    time_start = datetime.now()

    # 2.process entire mesh
    all_data = DualDataset.process_one_data(filename, opt.sub_size, filename_gt)
    centroid = all_data[0][0][0].centroid
    scale = all_data[0][0][0].scale
    print(F"---------- time: {(datetime.now()-time_start).total_seconds():7.4f} s")

    # 3.inference
    time_start1 = datetime.now()
    if len(all_data) == 1:
        dual_data = all_data[0][0]
        dual_data = DualDataset.post_processing(dual_data, opt.data_type)
        Vp, Np = predict_one_submesh(net, device, dual_data)
    else:
        sum_v = torch.zeros((mesh_noisy.n_vertices(), 1), dtype=torch.int8, device=device)
        Vp = torch.zeros((mesh_noisy.n_vertices(), 3), dtype=torch.float32, device=device)
        Np = torch.zeros((mesh_noisy.n_faces(), 3), dtype=torch.float32, device=device)

        for dual_data, V_idx, F_idx in all_data:
            dual_data = DualDataset.post_processing(dual_data, opt.data_type)
            vert_p, norm_p = predict_one_submesh(net, device, dual_data)
            sum_v[V_idx] += 1
            Vp[V_idx] += vert_p
            Np[F_idx] += norm_p

        Vp /= sum_v
        Np = torch.nn.functional.normalize(Np, dim=1)

    Vp = Vp.cpu() / scale + centroid
    Np = Np.cpu()

    # 4.update position and save
    fv_indices = torch.from_numpy(mesh_noisy.fv_indices()).long()
    vf_indices = torch.from_numpy(mesh_noisy.vf_indices()).long()
    depth_direction = None
    if opt.data_type in ['Kinect_v1', 'Kinect_v2']:
        depth_direction = torch.nn.functional.normalize(torch.from_numpy(points_noisy), dim=1)
    V = data_util.update_position2(Vp, fv_indices, vf_indices, Np, 60, depth_direction=depth_direction)
    om.write_mesh(F"{rst_filename[:-4]}-60.obj", om.TriMesh(V.numpy(), mesh_noisy.fv_indices()))

    print(F"---------- time: {(datetime.now()-time_start1).total_seconds():7.4f} s")
    
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
        data_dir = os.path.join(DATASET_DIR, opt.data_type, 'test')
        original_dir = os.path.join(DATASET_DIR, opt.data_type, 'test', 'original')
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
    net = network.DualGNN(force_depth=opt.force_depth, pool_type=opt.pool_type, wei_param=opt.wei_param)
    net.load_state_dict(torch.load(os.path.join(bak_dir, opt.model_name)))
    net = net.to(device)
    net.eval()

    # 3. infer
    error_all = np.zeros((3, len(filenames)))  # count, mean_error
    for i, noisy_file in enumerate(filenames):
        rst_file = os.path.join(result_dir, os.path.basename(noisy_file))
        # if os.path.exists(rst_file):
        #     continue
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

    opt.params_path = R"D:\code\python\3d\denoising\GeoBi-GNN\log\GeoBi-GNN_Kinect_Fusion_debug\20230314-141926\GeoBi-GNN_Kinect_Fusion_params.pth"
    # opt.data_dir = R"E:\code\python\denoising\TempNet\GanzangZhongliu\obj\noisy"
    # opt.data_dir = unicode(opt.data_dir, 'utf-8')
    opt.sub_size = 15000
    opt.gpu = 1

    predict_dir(opt.params_path, data_dir=opt.data_dir, sub_size=opt.sub_size, gpu=opt.gpu)

    # python test_dual.py --params_path=  --original_dir=  --gpu=  --sub_size=

    # download data from server via SSH
    # scp -r -P 25454 zyk@10.10.1.150:[source_path] [local_path]
