import os
import sys
import platform
import argparse
import random
from tqdm import tqdm
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter  # https://github.com/lanpa/tensorboard-pytorch
import network
import dataset

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


# ================================================= argparse ===================================================
# python train_dual.py --data_type=Synthetic --wei_type=0 --gpu= --flag=
# python train_dual.py --data_type=Kinect_v1 --wei_type=0 --gpu= --flag=
# python train_dual.py --data_type=Kinect_v2 --wei_type=0 --gpu= --flag=
# python train_dual.py --data_type=Kinect_Fusion --wei_type=0 --gpu= --flag=
def parse_arguments():
    parser = argparse.ArgumentParser()

    if is_debug():
        parser.add_argument('--data_type', type=str, default='Synthetic', help='Data type for training [default: Synthetic]')
        parser.add_argument('--flag', type=str, default='debug', help='Training flag [default: debug]')
        parser.add_argument('--gpu', type=int, default=-1, help='GPU to use')
    else:
        parser.add_argument('--data_type', type=str, required=True, help='Data type for training')
        parser.add_argument('--flag', type=str, required=True, help='Training flag')
        parser.add_argument('--gpu', type=int, required=True, help='GPU to use')

    parser.add_argument('--seed', type=int, default=None, help='Manual seed [default: None]')

    # data processing
    parser.add_argument('--train_txt', type=str, default=None, help='Training set list txt list')
    parser.add_argument('--test_txt', type=str, default='test_list-AA.txt', help='Test set list txt list')
    parser.add_argument('--filter_patch_count', type=int, default=100, help='')
    parser.add_argument('--sub_size', type=int, default=20000, help='The facet count of submesh if split big mesh [default: 20000]')

    parser.add_argument('--include_facet_initial_feature', action="store_true", help='whether include facet initial feature [default: False]')
    parser.add_argument('--wei_type', type=int, default=0, help='Edge weight type for Graclus graph pooling, 0:FGC, 1:guidance, 2:attention(softmax), 3:attention(sigmod)')
    parser.add_argument('--adaptive_loss_weight', action="store_true", help='whether adaptive learning loss weight [default: False]')

    parser.add_argument('--learn_res', action="store_true", help='whether learning the residual of target [default: True]')

    parser.add_argument('--loss_v', type=str, default='L1', help='vertex loss [default: L1]')
    parser.add_argument('--loss_n', type=str, default='L1', help='normal loss [default: L1]')
    parser.add_argument('--loss_v_scale', type=float, default=1, help='vertex loss scale [default: 1]')
    parser.add_argument('--loss_n_scale', type=float, default=1, help='normal loss scale [default: 1]')

    # training
    parser.add_argument('--max_epoch', type=int, default=500, help='Epoch to run [default: 50]')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
    parser.add_argument('--lr_sch', type=str, default='lmd', help='lr scheduler')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate [default: 0.0005]')
    parser.add_argument('--lr_step', type=int, nargs='+', default=[10], help='Decay step for learning rate [default: 2500]')
    parser.add_argument('--lr_decay', type=float, default=1, help='Decay rate for learning rate [default: 0.8]')
    parser.add_argument('--optimizer', type=str, default='adam', help='adam or sgd momentum [default: adam]')
    parser.add_argument('--momentum', type=float, default=0.9, help='Gradient descent momentum, for SGD [default: 0.9]')
    parser.add_argument('--beta1', type=float, default=0.9, help='First decay ratio, for Adam [default: 0.9]')
    parser.add_argument('--beta2', type=float, default=0.999, help='Second decay ratio, for Adam [default: 0.999]')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay coef [default: 0]')

    opt = parser.parse_args()

    if is_debug():
        opt.data_type = 'Kinect_v2'
        # opt.train_txt = 'train_list-AA.txt'
        opt.filter_patch_count = 100
        opt.sub_size = 20000
        opt.include_facet_initial_feature = True
        opt.wei_type = 0
        opt.adaptive_loss_weight = False

        opt.seed = 5752

        opt.learn_res = False
        opt.loss_v_scale = 1
        opt.loss_n_scale = 1

    opt.force_depth = True if opt.data_type in ['Kinect_v1', 'Kinect_v2'] else False
    opt.force_depth = False

    opt.pool_type = 'max'
    return opt


# ================================================== train =====================================================
def train(opt):
    assert (opt.loss_v in ['L1', 'L2', 'CD', 'EMD'])
    assert (opt.loss_n in ['L1', 'L2'])

    print('==='*30)
    training_name = F"Bi-GNN_{opt.data_type}"
    training_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    flag = opt.flag
    opt.flag = F"{training_name}_{flag}_{training_time}"
    print(F"Training flag: {opt.flag}")

    if opt.seed is None:
        opt.seed = random.randint(1, 10000)
    print(F"Random seed: {opt.seed}")
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    # ------------------------------------------------ 1.path ------------------------------------------------
    log_dir = os.path.join(LOG_DIR, F"{training_name}_{flag}", training_time)
    os.makedirs(log_dir, exist_ok=True)

    opt.model_name = F"{training_name}_model.pth"
    opt.params_name = F"{training_name}_params.pth"
    model_name = os.path.join(log_dir, opt.model_name)
    params_name = os.path.join(log_dir, opt.params_name)
    torch.save(opt, params_name)
    print('\n' + str(opt))

    sys_info = platform.system()
    if sys_info == "Windows":
        os.system(F"copy {os.path.join(BASE_DIR, 'train_dual.py')} {os.path.join(log_dir, 'train_dual.py')}")
        os.system(F"copy {os.path.join(BASE_DIR, 'network.py')} {os.path.join(log_dir, 'network.py')}")
        os.system(F"copy {os.path.join(BASE_DIR, 'dataset.py')} {os.path.join(log_dir, 'dataset.py')}")
    elif sys_info == "Linux":
        os.system(F"cp {os.path.join(BASE_DIR, 'train_dual.py')} {os.path.join(log_dir, 'train_dual.py')}")
        os.system(F"cp {os.path.join(BASE_DIR, 'network.py')} {os.path.join(log_dir, 'network.py')}")
        os.system(F"cp {os.path.join(BASE_DIR, 'dataset.py')} {os.path.join(log_dir, 'dataset.py')}")

    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    test_writer = SummaryWriter(os.path.join(log_dir, 'test'))
    test_writer.add_text('train_params', str(opt))
    info_name = os.path.join(log_dir, "training_info.txt")
    with open(info_name, 'w') as f:
        f.write(str(opt) + '\n')

    # ------------------------------------------------ 2.data ------------------------------------------------
    print('==='*30)
    train_dataset = dataset.DualDataset(opt.data_type, 'train', data_list_txt=opt.train_txt, weight_type=opt.wei_type, filter_patch_count=opt.filter_patch_count,
                                        submesh_size=opt.sub_size, transform=dataset.RandomRotate(False))
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=dataset.Collater([]))
    eval_dataset = dataset.DualDataset(opt.data_type, 'test', data_list_txt=opt.test_txt, weight_type=opt.wei_type, submesh_size=opt.sub_size)
    print(F"\nTraining set: {len(train_dataset):>4} samples")
    print(F"Testing set:  {len(eval_dataset):>4} samples")

    # ------------------------------------------------ 3.train ------------------------------------------------
    print('==='*30)
    # net = network.DualGNN(
    #     force_depth=opt.force_depth, include_facet_initial_feature=opt.include_facet_initial_feature, adaptive_loss_weight=opt.adaptive_loss_weight,
    #     pool_type=opt.pool_type, pool_step=2, edge_weight_type=opt.wei_type, learn_res=opt.learn_res)
    net = network.DualGNN_Fusion_temp(force_depth=opt.force_depth, pool_type=opt.pool_type, pool_step=2, edge_weight_type=opt.wei_type)

    total_params = sum(p.numel() for p in net.parameters())
    print('Total parameters: {}'.format(total_params))
    print('---'*30)

    device = torch.device(F"cuda:{opt.gpu}" if (opt.gpu >= 0 and torch.cuda.is_available()) else "cpu")
    net = net.to(device)

    # device = torch.device("cuda")
    # # os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    # # if opt.gpu is not None:
    # #     device_ids = [int(x) for x in opt.gpu.split(',')]
    # #     torch.backends.cudnn.benchmark = True
    # #     net.cuda(device_ids[0])
    # #     net = torch.nn.DataParallel(net, device_ids=device_ids)
    # # else:
    # #     net.cuda()
    # net = torch_geometric.nn.DataParallel(net).to(device)

    if opt.optimizer == 'sgd':
        optimizer = optim.SGD(net.parammeters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(net.parammeters(), lr=opt.lr, alpha=0.9)
    elif opt.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)

    if opt.lr_sch == 'step':
        lr_sch = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step[0], gamma=opt.lr_decay)
    elif opt.lr_sch == 'multi_step':
        lr_sch = lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_step, gamma=opt.lr_decay)
    elif opt.lr_sch == 'exp':
        lr_sch = lr_scheduler.ExponentialLR(optimizer, gamma=opt.lr_decay)
    elif opt.lr_sch == 'auto':
        lr_sch = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=5, verbose=True)
    else:
        def lmd_lr(step):
            return opt.lr_decay**(step/opt.lr_step[0])
        lr_sch = lr_scheduler.LambdaLR(optimizer, lr_lambda=lmd_lr)

    # -------------------------------------------- 4.loop over epochs --------------------------------------------
    print("Start training ...")
    time_start = datetime.now()
    best_error = float('inf')
    for epoch in range(1, opt.max_epoch):
        # training
        net.train()
        # l_bar='{desc}: {percentage:3.0f}%|'
        # r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        # bar = "{l_bar}{bar}{r_bar}"
        desc = "TRAINING - loss:{:.4f} {:.4f} {:.4f}  error:{:.4f} {:.4f}"
        bar = "{desc}  ({n_fmt}/{total_fmt}) [{elapsed}<{remaining}] {postfix}"
        pbar = tqdm(total=len(train_loader), ncols=90, leave=False, desc=desc.format(0, 0, 0, 0, 0), bar_format=bar)
        optimizer.zero_grad()
        for step, data in enumerate(train_loader):
            iteration = len(train_loader) * (epoch-1) + step
            data = [d.to(device) for d in data]

            vert_p, norm_p, alpha = net(data)
            train_loss_v = network.loss_v(vert_p, data[0].y, opt.loss_v)
            # train_loss_v_lap = network.laplacian_loss(vert_p, data[0].y, data[0].edge_index, data[0].normal)
            train_loss_f = network.loss_n(norm_p, data[1].y, opt.loss_n)
            # train_loss = network.dual_loss(train_loss_v, train_loss_f, v_scale=opt.loss_v_scale, n_scale=opt.loss_n_scale, alpha=alpha)
            train_loss = train_loss_v + train_loss_f
            train_error_v = network.error_v(vert_p, data[0].y)
            train_error_f = network.error_n(norm_p, data[1].y)

            # backward
            train_loss /= opt.batch_size
            train_loss.backward()
            train_loss *= opt.batch_size

            # gradient accumulation
            if (((step+1) % opt.batch_size) == 0) or (step+1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
                # log training information
                # last_lr = lr_sch.get_last_lr()[0]
                last_lr = optimizer.param_groups[0]['lr']
                train_writer.add_scalar('loss_v', train_loss_v.item(), iteration)
                train_writer.add_scalar('loss_f', train_loss_f.item(), iteration)
                train_writer.add_scalar('dual_loss', train_loss.item(), iteration)
                if alpha is not None:
                    train_writer.add_scalar('alpha', alpha.item(), iteration)
                train_writer.add_scalar('error_v', train_error_v.item(), iteration)
                train_writer.add_scalar('error_f', train_error_f.item(), iteration)

                pbar.desc = desc.format(train_loss_v, train_loss_f, train_loss, train_error_v, train_error_f)
                pbar.postfix = data[0].name.split('-')[0]
                pbar.update(opt.batch_size)
        pbar.close()
        lr_sch.step()
        # lr_sch.step(eval_error)

        # prediction
        net.eval()
        with torch.no_grad():
            desc = "VALIDATION - loss:{:.4f} {:.4f}  error:{:.4f} {:.4f}"
            pbar = tqdm(total=len(eval_dataset), ncols=90, leave=False, desc=desc.format(0, 0, 0, 0), bar_format=bar)
            eval_loss_v = eval_loss_f = eval_error_v = eval_error_f = count_v = count_f = 0
            for i, data in enumerate(eval_dataset):
                data = [d.to(device) for d in data]
                vert_p, norm_p, alpha_i = net(data)
                loss_i_v = network.loss_v(vert_p, data[0].y, opt.loss_v)
                # loss_i_v_lap = network.laplacian_loss(vert_p, data[0].y, data[0].edge_index)
                loss_i_f = network.loss_n(norm_p, data[1].y, opt.loss_n)
                error_i_v = network.error_v(vert_p, data[0].y)
                error_i_f = network.error_n(norm_p, data[1].y)

                eval_loss_v += loss_i_v * data[0].num_nodes
                eval_loss_f += loss_i_f * data[1].num_nodes
                eval_error_v += error_i_v * data[0].num_nodes
                eval_error_f += error_i_f * data[1].num_nodes
                count_v += data[0].num_nodes
                count_f += data[1].num_nodes

                pbar.desc = desc.format(loss_i_v, loss_i_f, error_i_v, error_i_f)
                pbar.postfix = data[0].name.split('-')[0]
                pbar.update(1)
            pbar.close()
            eval_loss_v /= count_v
            eval_loss_f /= count_f
            eval_error_v /= count_v
            eval_error_f /= count_f
            test_writer.add_scalar('loss_v', eval_loss_v.item(), iteration)
            test_writer.add_scalar('loss_f', eval_loss_f.item(), iteration)
            test_writer.add_scalar('error_v', eval_error_v.item(), iteration)
            test_writer.add_scalar('error_f', eval_error_f.item(), iteration)

        span = datetime.now() - time_start
        tqdm.write(F"Validation Results - {str(span).split('.')[0]:>8}  Epoch:{epoch:>3}  loss:{eval_loss_v:.4f} {eval_loss_f:.4f}  error:{eval_error_v:.4f} {eval_error_f:.4f}  lr:{last_lr:.4e}")

        # save model per epoch
        if eval_error_f < best_error:
            best_error = eval_error_f
            torch.save(net.state_dict(), model_name)

    train_writer.close()
    test_writer.close()
    print(F"\n{opt.flag}\nbest error: {best_error}")
    print('==='*30)

    return os.path.join(log_dir, params_name)


if __name__ == "__main__":
    opt = parse_arguments()

    params_file = train(opt)
    print("\n--- Training end ---")

    from test_dual import predict_dir
    predict_dir(params_file, data_dir=None, sub_size=opt.sub_size, gpu=opt.gpu)
