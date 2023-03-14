@echo off
call conda activate py37
@echo on

python test_dual.py --params_path=E:\code\python\denoising\TempNet\log\Bi-GNN_Kinect_v1_wei-type\20210520-170813\Bi-GNN_Kinect_v1_params.pth  --gpu=0 --data_dir=E:\data\dataset\Biwi_Kinect_Head_Pose-Database\02\mesh  --sub_size=200000
python test_dual.py --params_path=E:\code\python\denoising\TempNet\log\Bi-GNN_Kinect_v1_wei-type\20210525-071401\Bi-GNN_Kinect_v1_params.pth  --gpu=0 --data_dir=E:\data\dataset\Biwi_Kinect_Head_Pose-Database\02\mesh  --sub_size=200000


pause