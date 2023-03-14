import os, sys, shutil, random, argparse
from tqdm import tqdm
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter  # https://github.com/lanpa/tensorboard-pytorch
import network
import dataset


CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
LOG_DIR = os.path.join(BASE_DIR, 'log')
IS_DEBUG = getattr(sys, 'gettrace', None) is not None and sys.gettrace()


# Record the information printed in the terminal
class Print_Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()
        pass
# call by
# sys.stdout = Print_Logger(os.path.join(save_path,'test_log.txt'))


# ================================================= argparse ===================================================
# python train_dual.py  --data_type=Synthetic  --gpu=  --flag= 
def parse_arguments():
    parser = argparse.ArgumentParser()

    if IS_DEBUG:
        parser.add_argument('--data_type', type=str, default='Synthetic', help='Data type for training [default: Synthetic]')
        parser.add_argument('--flag', type=str, default='debug', help='Training flag [default: debug]')
        parser.add_argument('--gpu', type=int, default=-1, help='GPU to use')
    else:
        parser.add_argument('--data_type', type=str, required=True, help='Data type for training')
        parser.add_argument('--flag', type=str, required=True, help='Training flag')
        parser.add_argument('--gpu', type=int, required=True, help='GPU to use')

    parser.add_argument('--seed', type=int, default=None, help='Manual seed [default: None]')

    # data processing
    parser.add_argument('--filter_patch_count', type=int, default=100, help='submeshes that facet count less than this will not been included in training')
    parser.add_argument('--sub_size', type=int, default=20000, help='The facet count of submesh if split big mesh [default: 20000]')

    parser.add_argument('--loss_v', type=str, default='L1', help='vertex loss [default: L1]')
    parser.add_argument('--loss_n', type=str, default='L1', help='normal loss [default: L1]')
    parser.add_argument('--loss_v_scale', type=float, default=1, help='vertex loss scale [default: 1]')
    parser.add_argument('--loss_n_scale', type=float, default=1, help='normal loss scale [default: 1]')

    parser.add_argument('--wei_param', type=int, default=2)

    # training
    parser.add_argument('--max_epoch', type=int, default=1000, help='Epoch to run [default: 1000]')
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

    parser.add_argument('--restore', action="store_true", help='')
    parser.add_argument('--model_path', type=str, default=None, help='')

    # opt = parser.parse_args()
    opt, ext_list = parser.parse_known_args()

    to_dic = lambda kvlist: eval(F" {{ '{kvlist[0]}': {kvlist[1]} }} ")  # for extent argparse
    for arg in ext_list:
        opt.__dict__.update(to_dic(arg[2:].split('=', 1)))
        pass

    if IS_DEBUG:
        opt.data_type = 'Kinect_v1'
        opt.sub_size = 20000
        opt.wei_param = 10

    opt.force_depth = True if opt.data_type in ['Kinect_v1', 'Kinect_v2'] else False
    opt.pool_type = 'max'

    return opt


# ================================================== train =====================================================
def train(opt):

    print('==='*30)
    training_name = F"GeoBi-GNN_{opt.data_type}"
    training_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    flag = opt.flag
    opt.flag = F"{training_name}_{flag}_{training_time}"
    print(F"Training flag: {opt.flag}")

    # seed
    if opt.seed is None:
        opt.seed = random.randint(1, 10000)
    print(F"Random seed: {opt.seed} \n")
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    # ------------------------------------------------ 1.path ------------------------------------------------
    log_dir = os.path.join(LOG_DIR, F"{training_name}_{flag}", training_time)
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Print_Logger(os.path.join(log_dir,'training_info.txt'))

    opt.model_name = F"{training_name}_model.pth"
    opt.params_name = F"{training_name}_params.pth"
    model_name = os.path.join(log_dir, opt.model_name)
    params_name = os.path.join(log_dir, opt.params_name)
    torch.save(opt, params_name)
    print(str(opt))

    # backup code
    shutil.copytree(CODE_DIR, os.path.join(log_dir, 'code_bak'))

    # tensorboard
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    test_writer = SummaryWriter(os.path.join(log_dir, 'test'))
    test_writer.add_text('train_params', str(opt))

    # ------------------------------------------------ 2.data ------------------------------------------------
    print('==='*30)
    train_dataset = dataset.DualDataset(opt.data_type, 'train', data_list_txt='train_list.txt', filter_patch_count=opt.filter_patch_count,
                                        submesh_size=opt.sub_size, transform=dataset.RandomRotate(False))
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=dataset.Collater([]))
    eval_dataset = dataset.DualDataset(opt.data_type, 'test', data_list_txt='test_list.txt', submesh_size=opt.sub_size)
    print(F"\nTraining set: {len(train_dataset):>4} samples")
    print(F"Testing set:  {len(eval_dataset):>4} samples")

    # ------------------------------------------------ 3.train ------------------------------------------------
    print('==='*30)
    net = network.DualGNN(force_depth=opt.force_depth, pool_type=opt.pool_type, edge_weight_type=10, wei_param=opt.wei_param)

    total_params = sum(p.numel() for p in net.parameters())
    print('Total parameters: {}'.format(total_params))
    print('---'*30)

    device = torch.device(F"cuda:{opt.gpu}" if (opt.gpu >= 0 and torch.cuda.is_available()) else "cpu")
    last_epoch = 0
    if opt.restore:
        net.load_state_dict(torch.load(opt.model_path))
        last_epoch = 500
    net = net.to(device)

    if opt.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=opt.lr, alpha=0.9)
    elif opt.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)

    if opt.lr_sch == 'step':
        lr_sch = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step[0], gamma=opt.lr_decay)
    elif opt.lr_sch == 'multi_step':
        lr_sch = lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_step, gamma=opt.lr_decay)
    elif opt.lr_sch == 'exp':
        lr_sch = lr_scheduler.ExponentialLR(optimizer, gamma=opt.lr_decay)
    elif opt.lr_sch == 'auto':
        lr_sch = lr_scheduler.ReduceLROnPlateau(optimizer, factor=opt.lr_decay, patience=opt.lr_step[0], verbose=True)
    else:
        def lmd_lr(step):
            return opt.lr_decay**(step/opt.lr_step[0])
        lr_sch = lr_scheduler.LambdaLR(optimizer, lr_lambda=lmd_lr)

    # -------------------------------------------- 4.loop over epochs --------------------------------------------
    print("Start training ...")
    time_start = datetime.now()
    best_error = float('inf')

    for epoch in range(last_epoch, opt.max_epoch):
        print_log = epoch % 10 == 0

        # training
        net.train()
        # l_bar='{desc}: {percentage:3.0f}%|'
        # r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        # bar = "{l_bar}{bar}{r_bar}"
        desc = "TRAINING - epoch:{:>3} loss:{:.4f} {:.4f} {:.4f}  error:{:.4f} {:.4f}"
        bar = "{desc}  ({n_fmt}/{total_fmt}) [{elapsed}<{remaining}]"
        pbar = tqdm(total=len(train_loader), ncols=90, leave=False, desc=desc.format(epoch, 0, 0, 0, 0, 0), bar_format=bar)
        optimizer.zero_grad()
        for step, data in enumerate(train_loader):
            iteration = len(train_loader) * (epoch-1) + step
            data = [d.to(device) for d in data]

            vert_p, norm_p, _ = net(data)
            train_loss_v = network.loss_v(vert_p, data[0].y, opt.loss_v)
            train_loss_f = network.loss_n(norm_p, data[1].y, opt.loss_n)
            train_loss = network.dual_loss(train_loss_v, train_loss_f, v_scale=opt.loss_v_scale, n_scale=opt.loss_n_scale)
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
                train_writer.add_scalar('error_v', train_error_v.item(), iteration)
                train_writer.add_scalar('error_f', train_error_f.item(), iteration)

                pbar.desc = desc.format(epoch, train_loss_v, train_loss_f, train_loss, train_error_v, train_error_f)
                pbar.update(opt.batch_size)
        pbar.close()

        # prediction
        net.eval()
        with torch.no_grad():
            desc = "VALIDATION - epoch:{:>3} loss:{:.4f} {:.4f}  error:{:.4f} {:.4f}"
            pbar = tqdm(total=len(eval_dataset), ncols=90, leave=False, desc=desc.format(epoch, 0, 0, 0, 0), bar_format=bar)
            eval_loss_v = eval_loss_f = eval_error_v = eval_error_f = count_v = count_f = 0
            for i, data in enumerate(eval_dataset):
                data = [d.to(device) for d in data]
                vert_p, norm_p, _ = net(data)
                loss_i_v = network.loss_v(vert_p, data[0].y, opt.loss_v)
                loss_i_f = network.loss_n(norm_p, data[1].y, opt.loss_n)
                error_i_v = network.error_v(vert_p, data[0].y)
                error_i_f = network.error_n(norm_p, data[1].y)

                eval_loss_v += loss_i_v * data[0].num_nodes
                eval_loss_f += loss_i_f * data[1].num_nodes
                eval_error_v += error_i_v * data[0].num_nodes
                eval_error_f += error_i_f * data[1].num_nodes
                count_v += data[0].num_nodes
                count_f += data[1].num_nodes

                pbar.desc = desc.format(epoch, loss_i_v, loss_i_f, error_i_v, error_i_f)
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

        if opt.lr_sch == 'auto':
            lr_sch.step(eval_error_f)
        else:
            lr_sch.step()

        span = datetime.now() - time_start
        str_log = F"Epoch {epoch:>3}: {str(span).split('.')[0]:>8}  loss:{eval_loss_v:.4f} {eval_loss_f:.4f} | "
        str_log += F"error:{eval_error_v:.4f} {eval_error_f:.4f}  lr:{last_lr:.4e}"
        # save model per epoch
        if eval_error_f < best_error:
            best_error = eval_error_f
            torch.save(net.state_dict(), model_name)
            str_log = str_log + " - save model"
            print_log = True

        if print_log:
            tqdm.write(str_log)

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

    input_re = [0, 1, 2, 3]
    lr = [0.005, 0.002, 0.001, 0.0005]
    batch_size = [2, 4, 5]

    lr_step = [50, 60, 80]
    lr_decay = [0.8, 0.5, 0.2, 0.1]
    wei_decay = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
