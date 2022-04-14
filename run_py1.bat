@echo off
call conda activate py37
@echo on

python test_dual.py --params_path=E:\code\python\denoising\TempNet\log\Bi-GNN_Synthetic_wei-type\20210510-110214\Bi-GNN_Synthetic_params.pth  --gpu=1 --data_dir=E:\data\dataset\mesh_denoising_data\Synthetic\noisy_0.6
python test_dual.py --params_path=E:\code\python\denoising\TempNet\log\Bi-GNN_Synthetic_wei-type\20210510-110708\Bi-GNN_Synthetic_params.pth  --gpu=1 --data_dir=E:\data\dataset\mesh_denoising_data\Synthetic\noisy_0.6
python test_dual.py --params_path=E:\code\python\denoising\TempNet\log\Bi-GNN_Synthetic_wei-type\20210512-011022\Bi-GNN_Synthetic_params.pth  --gpu=1 --data_dir=E:\data\dataset\mesh_denoising_data\Synthetic\noisy_0.6
python test_dual.py --params_path=E:\code\python\denoising\TempNet\log\Bi-GNN_Synthetic_wei-type\20210512-031208\Bi-GNN_Synthetic_params.pth  --gpu=1 --data_dir=E:\data\dataset\mesh_denoising_data\Synthetic\noisy_0.6

python test_dual.py --params_path=E:\code\python\denoising\TempNet\log\Bi-GNN_Synthetic_wei-type\20210510-110214\Bi-GNN_Synthetic_params.pth  --gpu=1 --data_dir=E:\data\dataset\mesh_denoising_data\Synthetic\noisy_0.7
python test_dual.py --params_path=E:\code\python\denoising\TempNet\log\Bi-GNN_Synthetic_wei-type\20210510-110708\Bi-GNN_Synthetic_params.pth  --gpu=1 --data_dir=E:\data\dataset\mesh_denoising_data\Synthetic\noisy_0.7
python test_dual.py --params_path=E:\code\python\denoising\TempNet\log\Bi-GNN_Synthetic_wei-type\20210512-011022\Bi-GNN_Synthetic_params.pth  --gpu=1 --data_dir=E:\data\dataset\mesh_denoising_data\Synthetic\noisy_0.7
python test_dual.py --params_path=E:\code\python\denoising\TempNet\log\Bi-GNN_Synthetic_wei-type\20210512-031208\Bi-GNN_Synthetic_params.pth  --gpu=1 --data_dir=E:\data\dataset\mesh_denoising_data\Synthetic\noisy_0.7

pause