@echo off
call conda activate py37

set t=3000
echo sleep %t%s ...
ping -n %t% 127.0.0.1>null
echo sleep end

@echo on


python train_dual.py --data_type=Kinect_v2 --gpu=0 --flag=init-lr --wei_type=3 --include_facet_initial_feature --learn_res --lr_sch=auto --lr_decay=0.8 --lr_step=10
python train_dual.py --data_type=Kinect_v2 --gpu=0 --flag=init-lr --wei_type=3 --include_facet_initial_feature --learn_res --lr_sch=auto --lr_decay=0.6 --lr_step=10

python train_dual.py --data_type=Kinect_v2 --gpu=0 --flag=init-lr --wei_type=3 --include_facet_initial_feature --learn_res --lr_sch=auto --lr_decay=0.6 --lr_step=10 --AWA --T=15
python train_dual.py --data_type=Kinect_v2 --gpu=0 --flag=init-lr --wei_type=3 --include_facet_initial_feature --learn_res --lr_sch=auto --lr_decay=0.6 --lr_step=10 --AWA --T=20


pause