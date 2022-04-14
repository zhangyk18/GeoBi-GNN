import os
import sys
import glob
from tqdm import tqdm
import numpy as np
import openmesh as om
import data_util
import torch
import torch_geometric
from torch_geometric.utils import to_undirected, add_self_loops
from torch_geometric.data import Data, Batch
import torch_geometric.nn.pool.voxel_grid
# from torch_geometric.utils import remove_self_loops, add_remaining_self_loops, softmax, geodesic_distance, degree
# from torch_geometric.transforms import SamplePoints, GridSampling
# from torch_geometric.data import Dataset, DataLoader, ClusterData

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'log')


class Collater(object):
    def __init__(self, follow_batch):
        self.follow_batch = follow_batch

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, Data):
            return Batch.from_data_list(batch, self.follow_batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, tuple):
            # return type(elem)(self.collate(s) for s in zip(*batch))
            return elem

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)


class RandomRotate(object):
    def __init__(self, z_rotated=True):
        self.z_rotated = z_rotated

    def __call__(self, data):
        # rotation
        angles = np.random.uniform(size=(3)) * 2 * np.pi
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        rotation_matrix = Rz if self.z_rotated else np.dot(Rz, np.dot(Ry, Rx))
        rotation_matrix = torch.from_numpy(rotation_matrix).to(data[0].y.dtype).to(data[0].y.device)

        for d in data:
            d.x[:, 0:3] = torch.matmul(d.x[:, 0:3], rotation_matrix)
            d.x[:, 3:6] = torch.matmul(d.x[:, 3:6], rotation_matrix)
            d.y[:, 0:3] = torch.matmul(d.y[:, 0:3], rotation_matrix)
            if hasattr(d, 'pos') and d.pos is not None:
                d.pos = torch.matmul(d.pos, rotation_matrix)
            if hasattr(d, 'centroid') and d.centroid is not None:
                d.centroid = torch.matmul(d.centroid, rotation_matrix)
            if hasattr(d, 'depth_direction') and d.depth_direction is not None:
                d.depth_direction = torch.matmul(d.depth_direction, rotation_matrix)

        return data


class DualDataset(torch_geometric.data.Dataset):
    def __init__(self, data_type, train_or_test='train', data_list_txt=None,
                 filter_patch_count=0, submesh_size=sys.maxsize, transform=None):
        self.data_type = data_type
        self.root_dir = os.path.join(DATA_DIR, data_type)
        self.data_dir = os.path.join(DATA_DIR, data_type, train_or_test)  # train or test
        self.filter_patch_count = filter_patch_count
        self.submesh_size = submesh_size
        self.processed_folder = "processed_data"
        super(DualDataset, self).__init__(transform=transform)

        # 1.data list paths
        self.processed_files = []
        self.files_noisy = []
        self.files_original = []
        noisy_dir = os.path.join(self.data_dir, 'noisy')
        original_dir = os.path.join(self.data_dir, 'original')
        if data_list_txt is not None:
            data_list = list(filter(None, [line.strip() for line in open(os.path.join(self.root_dir, data_list_txt))]))
        else:
            data_list = glob.glob(os.path.join(original_dir, "*.obj"))
            data_list = [os.path.basename(d)[:-4] for d in data_list]
        for name in data_list:
            files_n = glob.glob(os.path.join(noisy_dir, F"{name}_n*.obj"))
            for name_n in files_n:
                self.files_noisy.append(name_n)
                self.files_original.append(os.path.join(original_dir, F"{name}.obj"))

        # 2.data preprocess
        self.process_data()

    @property
    def processed_dir(self):
        return os.path.join(self.data_dir, self.processed_folder)

    def process_data(self):
        print('Processing...')

        os.makedirs(self.processed_dir, exist_ok=True)
        # l_bar='{desc}: {percentage:3.0f}%|'
        # r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        # bar = "{l_bar}{bar}{r_bar}"
        bar = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
        pbar = tqdm(total=len(self.files_noisy), ncols=85, desc="Data list info: ", bar_format=bar)
        for _, (noisyfile, originalfile) in enumerate(zip(self.files_noisy, self.files_original)):
            pbar.postfix = os.path.basename(noisyfile)[:-4]
            pbar.update(1)
            self.process_one_data(noisyfile, self.submesh_size, originalfile=originalfile, obj=self)
        pbar.close()

        print('Done!')

    @staticmethod
    def process_one_data(noisyfile, submesh_size, originalfile=None, obj=None):
        filter_patch_count = 0 if obj is None else obj.filter_patch_count

        # 1.read data
        mesh_noisy = om.read_trimesh(noisyfile)
        mesh_original = None if originalfile is None else om.read_trimesh(originalfile)
        points_noisy = mesh_noisy.points().astype(np.float32)
        points_original = None if mesh_original is None else mesh_original.points().astype(np.float32)

        # 2.center and scale, record
        _, centroid, scale = data_util.center_and_scale(points_noisy, mesh_noisy.ev_indices())

        # 3.split to submeshes
        all_dual_data = []
        if mesh_noisy.n_faces() <= submesh_size:
            filename = os.path.basename(noisyfile)[:-4]
            if obj is not None:
                pro_name = os.path.join(obj.processed_dir, F"{filename}.pt")
                obj.processed_files.append(pro_name)
            if obj is None or not os.path.exists(pro_name):
                dual_data = DualDataset.process_one_submesh(mesh_noisy, filename, mesh_original)
                dual_data[0].centroid = torch.from_numpy(centroid).float()
                dual_data[0].scale = scale
                all_dual_data.append((dual_data, None, None))
            if obj is not None and not os.path.exists(pro_name):
                torch.save(dual_data, pro_name)
        else:
            flag = np.zeros(mesh_noisy.n_faces(), dtype=np.bool)

            fv_indices = mesh_noisy.fv_indices()
            vf_indices = mesh_noisy.vf_indices()
            face_cent = points_noisy[fv_indices].mean(1)
            seed = np.argmax(((face_cent - centroid)**2).sum(1))
            for sub_num in range(1, sys.maxsize):
                # select patch facet indices
                select_faces = data_util.mesh_get_neighbor_np(fv_indices, vf_indices, seed, neighbor_count=submesh_size)
                flag.put(select_faces, True)

                if len(select_faces) > filter_patch_count:
                    filename = F"{os.path.basename(noisyfile)[:-4]}-sub{submesh_size}-{seed}"
                    if obj is not None:
                        pro_name = os.path.join(obj.processed_dir, F"{filename}.pt")
                        obj.processed_files.append(pro_name)
                    if obj is None or not os.path.exists(pro_name):
                        # split submesh based on facet indices
                        V_idx, F = data_util.get_submesh(fv_indices, select_faces)
                        submesh_n = om.TriMesh(points_noisy[V_idx], F)
                        submesh_o = None if points_original is None else om.TriMesh(points_original[V_idx], F)
                        dual_data = DualDataset.process_one_submesh(submesh_n, filename, submesh_o)
                        dual_data[0].centroid = torch.from_numpy(centroid).float()
                        dual_data[0].scale = scale
                        all_dual_data.append((dual_data, V_idx, select_faces))
                    if obj is not None and not os.path.exists(pro_name):
                        torch.save(dual_data, pro_name)
                        # save for visualization
                        om.write_mesh(os.path.join(obj.processed_dir, F"{filename}.obj"), submesh_n)

                # whether all facets have been visited, next seed
                left_idx = np.where(~flag)[0]
                if left_idx.size:
                    idx_temp = np.argmax(((face_cent[left_idx] - centroid)**2).sum(1))
                    seed = left_idx[idx_temp]
                else:  # all have been visited
                    break
        return all_dual_data

    @staticmethod
    def process_one_submesh(mesh_n, name='graph', mesh_o=None):
        mesh_n.update_face_normals()
        mesh_n.update_vertex_normals()

        ev_indices = torch.from_numpy(mesh_n.ev_indices()).long()
        fv_indices = torch.from_numpy(mesh_n.fv_indices()).long()
        vf_indices = torch.from_numpy(mesh_n.vf_indices()).long()
        vv_indices = torch.from_numpy(mesh_n.vv_indices()).long()

        edge_dual_fv = data_util.build_edge_fv(fv_indices)

        # vertex graph
        pos_v = torch.from_numpy(mesh_n.points()).float()
        normal_v = torch.from_numpy(mesh_n.vertex_normals()).float()
        edge_idx_v = ev_indices.T  # directed grah, no self_loops
        edge_idx_v = to_undirected(edge_idx_v)
        edge_idx_v, _ = add_self_loops(edge_idx_v)
        # edge_idx_v = data_util.build_vertex_graph(ev_indices, vv_indices)  # undirected grah, with self_loops
        edge_wei_v = data_util.calc_weight(pos_v, normal_v, edge_idx_v)
        depth_direction = torch.nn.functional.normalize(pos_v, dim=1).float()  # unit vector
        graph_v = Data(name=F"{name}-v",
                       pos=pos_v.float(), normal=normal_v.float(), edge_index=edge_idx_v, edge_weight=edge_wei_v.float(),
                       depth_direction=depth_direction, edge_dual=edge_dual_fv[1])

        # facet graph
        pos_f = pos_v[fv_indices].mean(1).float()
        normal_f = torch.from_numpy(mesh_n.face_normals()).float()
        normal_f = normal_f.reshape(-1, 3)
        edge_idx_f = data_util.build_facet_graph(fv_indices, vf_indices)  # undirected grah, with self_loops
        edge_wei_f = data_util.calc_weight(pos_f, normal_f, edge_idx_f)
        graph_f = Data(name=F"{name}-f",
                       pos=pos_f.float(), normal=normal_f.float(), edge_index=edge_idx_f, edge_weight=edge_wei_f.float(),
                       fv_indices=fv_indices, edge_dual=edge_dual_fv[0])

        # _, centroid, scale = data_util.center_and_scale(pos_v, ev_indices, 0)
        # graph_v.centroid = centroid.float()
        # graph_v.scale = scale

        if mesh_o is not None:
            mesh_o.update_face_normals()
            mesh_o.update_vertex_normals()
            # vertex graph
            graph_v.y = torch.from_numpy(mesh_o.points()).float()
            # facet graph
            graph_f.y = torch.from_numpy(mesh_o.face_normals()).float()

        return (graph_v, graph_f)

    @staticmethod
    def post_processing(dual_data, data_type, is_plot=False):
        data_v, data_f = dual_data

        # facet graph
        f_pos_normalized = (data_f.pos - data_v.centroid) * data_v.scale
        data_f.x = torch.cat((f_pos_normalized, data_f.normal), 1)
        data_f.normal = data_f.edge_dual = None
        if not is_plot:
            data_f.pos = None

        # vertex graph
        v_pos_normalized = (data_v.pos - data_v.centroid) * data_v.scale
        data_v.x = torch.cat((v_pos_normalized, data_v.normal), 1)
        data_v.y = None if data_v.y is None else (data_v.y - data_v.centroid) * data_v.scale
        data_v.normal = data_v.centroid = data_v.scale = data_v.edge_dual = None
        if not is_plot:
            data_v.pos = None
        else:
            data_v.pos = data_v.y
            data_v.fv_indices = data_f.fv_indices

        if data_type not in ['Kinect_v1', 'Kinect_v2']:
            data_v.depth_direction = None
        return data_v, data_f

    def len(self):
        return len(self.processed_files)

    def get(self, idx):
        dual_data = torch.load(self.processed_files[idx])
        return DualDataset.post_processing(dual_data, self.data_type)


class MultiDataset(torch_geometric.data.Dataset):
    def __init__(self, dataset_list):
        super(MultiDataset, self).__init__()
        self.dataset_list = dataset_list
        self.dataset_start = []
        self.dataset_count = []
        for ds in dataset_list:
            if len(self.dataset_start) > 0:
                self.dataset_start.append(self.dataset_start[-1] + self.dataset_count[-1])
            else:
                self.dataset_start.append(0)
            self.dataset_count.append(len(ds))

    def len(self):
        return sum(self.dataset_count)

    def get(self, idx):
        dataset_idx, item_idx = self.get_idx(idx)
        return self.dataset_list[dataset_idx].get(item_idx)

    def get_idx(self, idx):
        for data_idx, start in enumerate(self.dataset_start):
            if idx < start:
                data_idx -= 1
                break
        return data_idx, idx - self.dataset_start[data_idx]


def process_GT_Kinect_Fusion(noisy_dir, original_dir, filtered_dir):
    result_dir = os.path.join(filtered_dir, 'GT_file')
    os.makedirs(result_dir, exist_ok=True)

    noisy_files = []
    original_files = []
    filtered_files = []
    data_list = [os.path.basename(d)[:-4] for d in glob.glob(os.path.join(original_dir, "*.obj"))]
    for name in data_list:
        files_n = glob.glob(os.path.join(noisy_dir, F"{name}*.obj"))
        files_f = glob.glob(os.path.join(filtered_dir, F"{name}*.obj"))
        for i, name_n in enumerate(files_n):
            noisy_files.append(name_n)
            original_files.append(os.path.join(original_dir, F"{name}.obj"))
            filtered_files.append(files_f[i])

    for _, (noisyfile, originalfile, filteredfile) in enumerate(zip(noisy_files, original_files, filtered_files)):
        mesh_n = om.read_trimesh(noisyfile)
        mesh_n.update_face_normals()
        normal1 = mesh_n.face_normals()

        mesh_o = om.read_trimesh(originalfile)
        mesh_o.update_face_normals()
        normal2 = mesh_o.face_normals()

        mesh_f = om.read_trimesh(filteredfile)
        mesh_f.update_face_normals()
        normal3 = mesh_f.face_normals()

        for i in range(normal1.shape[0]):
            mesh_f.set_color(mesh_f.face_handle(i), np.concatenate(((normal1[i]+1)/2.0, [1])))
        rst_file = os.path.join(result_dir, F"{os.path.basename(noisyfile)[:-4]}-color_n.off")
        om.write_mesh(rst_file, mesh_f, face_color=True)

        fv_indices = mesh_n.fv_indices()
        vf_indices = mesh_n.vf_indices()
        for i in range(normal1.shape[0]):
            faces_ring2 = data_util.mesh_get_neighbor_np(fv_indices, vf_indices, i, ring_count=2)
            normal_ring2 = normal2[faces_ring2]
            error = ((normal_ring2-normal3[i])**2).sum(1)
            norm_t = normal_ring2[np.argmin(error)]
            mesh_f.set_color(mesh_f.face_handle(i), np.concatenate(((norm_t+1)/2.0, [1])))
        rst_file = os.path.join(result_dir, F"{os.path.basename(noisyfile)[:-4]}-color_f.off")
        om.write_mesh(rst_file, mesh_f, face_color=True)

        for i in range(normal1.shape[0]):
            mesh_f.set_color(mesh_f.face_handle(i), np.concatenate(((normal2[i]+1)/2.0, [1])))
        rst_file = os.path.join(result_dir, F"{os.path.basename(noisyfile)[:-4]}-color_o.off")
        om.write_mesh(rst_file, mesh_f, face_color=True)

        # for i in range(normal1.shape[0]):
        #     mesh_n.set_normal(mesh_f.face_handle(i), normal[i])
        # rst_file = os.path.join(result_dir, F"{os.path.basename(noisyfile)[:-4]}-normal.stl")
        # om.write_mesh(rst_file, mesh_f, face_normal=True)

        print(F"{rst_file} saved \n")

    print("--- END ---")


if __name__ == "__main__":

    torch.manual_seed(1)

    train_dataset = DualDataset("Kinect_v2", 'train')
    for i, data in enumerate(train_dataset):
        pass
        if isinstance(data, Data):
            print(len(data), end=', ')
        elif isinstance(data, tuple):
            # print(F"{data[0]['name']}:{data[0].num_nodes}, {data[1]['name']}:{data[1].num_nodes}")
            data = data[0]
            x = data.x[:, :3]
            y = data.y[:, :3]
            err = x - y
            err = (err**2).sum(1)**0.5
            print(F"{data['name']}:{data.num_nodes}, err_mean:{err.mean():.6f}, err_max:{err.max():.6f}, err_min:{err.min():.6f}")


    noisy_dir = R"E:\data\dataset\mesh_denoising_data\Kinect_Fusion\test\noisy"
    original_dir = R"E:\data\dataset\mesh_denoising_data\Kinect_Fusion\test\original"
    filtered_dir = R"E:\data\dataset\mesh_denoising_data\Kinect_Fusion\test\result_GNF"
    process_GT_Kinect_Fusion(noisy_dir, original_dir, filtered_dir)
    exit()


    ts_pkl = R"E:\code\python\Facet_Graph_Convolution\Preprocessed_Data\trainingSet.pkl"
    vs_pkl = R"E:\code\python\Facet_Graph_Convolution\Preprocessed_Data\validSet.pkl"
    # dataset = RandomSplitDataset('Synthetic', "train_list-AA.txt", submesh_size=50000, transform=RandomRotate(False))
    # dataset = DualDataset('Synthetic', data_list_txt="train_list-AA.txt", submesh_size=20000, transform=RandomRotate(False))
    dataset1 = DualDataset('Kinect_v2', 'test', submesh_size=20000, filter_patch_count=100, transform=RandomRotate(False))
    dataset3 = DualDataset('Kinect_v2', 'train', data_list_txt="train_list-AA.txt", submesh_size=100000, transform=RandomRotate(False))

    two_dataset = MultiDataset([dataset1, dataset2, dataset1, dataset2])

    for i, data in enumerate(two_dataset):
        pass
        if isinstance(data, Data):
            print(len(data), end=', ')
        elif isinstance(data, tuple):
            # print(F"{data[0]['name']}:{data[0].num_nodes}, {data[1]['name']}:{data[1].num_nodes}")
            data = data[0]
            x = data.x[:, :3]
            y = data.y[:, :3]
            err = x - y
            err = (err**2).sum(1)**0.5
            print(F"{data['name']}:{data.num_nodes}, err_mean:{err.mean():.6f}, err_max:{err.max():.6f}, err_min:{err.min():.6f}")



    train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=Collater([]))
    for epoch in range(2):
        for idx, data in enumerate(train_loader):
            if isinstance(data, Batch):
                print(len(data), end=', ')
            elif isinstance(data, tuple):
                print(F"{data[0]['name']}:{data[0].num_nodes}, {data[1]['name']}:{data[1].num_nodes}")
                data_v = data[0]
                data_f = data[1]
                pass

    # data_fp = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'ShapeNet')
    # category = ['Airplane', 'Bag']
    # train_dataset = ShapeNet(data_fp, category)
    # train_dataset.shuffle()
    # train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    # for idx, data in enumerate(train_loader):
    #     print("---")
    #     print(data.num_nodes)

    # obj_file = R"E:\code\python\denoising\TempNet\data\Synthetic\noisy\plane-sphere-n1.obj"
    # mesh = MeshGraph(obj_file)
    # face_patch_idx = mesh.get_face_patch(10194, 50)

    # print("--- dataset end ---")
