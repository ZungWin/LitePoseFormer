import os
import glob
import torch
import random
import logging
import matplotlib
import numpy as np
matplotlib.use('Agg')
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from common.graph_utils import adj_mx_from_skeleton, adj_mx_from_skeleton_temporal
from demo.vis import show3Dpose, show2Dpose
from common.utils import *
from common.camera import *
import common.eval_cal as eval_cal
from common.arguments import parse_args
from common.camera import *
from common.load_data_hm36 import Fusion
from common.load_data_3dhp import Fusion_3dhp
from common.h36m_dataset import Human36mDataset
from common.mpi_inf_3dhp_dataset import Mpi_inf_3dhp_Dataset
from model.graphtoken import Model
import cv2
import copy
args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def train(dataloader, model, optimizer, epoch):
    model.train()
    loss_all = {'loss': AccumLoss()}

    stride_num = [9, 3, 3]

    w_1 = nn.Conv1d(2, 2, kernel_size=1, stride=1).cuda()
    w_2 = [nn.Conv1d(2, 2, kernel_size=3, stride=stride_num[i], padding = 1).cuda() for i in range(len(stride_num))]
    for i, data in enumerate(tqdm(dataloader, 0)):
        batch_cam, gt_3D, input_2D, input_2D_GT, action, subject, cam_ind = data
        [input_2D, input_2D_GT, gt_3D, batch_cam] = get_varialbe('train', [input_2D, input_2D_GT, gt_3D, batch_cam])
        # input_2D: 8,1,17,2
        output_3D = model(input_2D)  # bs,1,17,3
       
        out_target = gt_3D.clone()  # bs,243,17,3
        out_target[:, :, 0, :] = 0

        fig = plt.figure(figsize=(9.6, 5.4))
        gs = gridspec.GridSpec(1, 1)
        gs.update(wspace=-0.00, hspace=0.05)
        ax = plt.subplot(gs[0], projection='3d')
        for b in range(input_2D_GT.shape[0]):
            output_dir_pose_b = os.path.join(output_dir_pose, f'{b}')
            os.makedirs(output_dir_pose_b, exist_ok=True)
            joint = out_target[0,4].cpu().detach().numpy()
            rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
            rot = np.array(rot, dtype='float32')
            post_out = camera_to_world(joint, R=rot, t=0)  # 世界坐标系
            post_out[:, 2] -= np.min(post_out[:, 2])
            show3Dpose(post_out, ax, True)
            output_image_path = os.path.join(output_dir_pose_b, f'3D.png')
            plt.savefig(output_image_path, dpi=200, format='png', bbox_inches='tight')
        
        out_target[:, :, args.root_joint] = 0  # bs,243,17,3
        out_target = out_target[:, args.pad].unsqueeze(1) # bs,1,17,3

        loss = eval_cal.mpjpe(output_3D, out_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        N = input_2D.shape[0]
        loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

    return loss_all['loss'].avg


def test(actions, dataloader, model):
    model.eval()

    action_error = define_error_list(actions)

    for i, data in enumerate(tqdm(dataloader, 0)):
        batch_cam, gt_3D, input_2D, input_2D_GT, action, subject, cam_ind = data
        [input_2D, input_2D_GT, gt_3D, batch_cam] = get_varialbe('test', [input_2D, input_2D_GT, gt_3D, batch_cam])

        output_3D_non_flip = model(input_2D[:, 0])
        output_3D_flip     = model(input_2D[:, 1])

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, args.joints_left + args.joints_right, :] = output_3D_flip[:, :, args.joints_right + args.joints_left, :] 

        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        out_target = gt_3D.clone()
        out_target = out_target[:, args.pad].unsqueeze(1)

        output_3D[:, :, args.root_joint] = 0
        out_target[:, :, args.root_joint] = 0
        rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
        rot = np.array(rot, dtype='float32')
        output_dir = './'
        output_dir_3D = output_dir +'pose3D/'
        os.makedirs(output_dir_3D, exist_ok=True)
        for b in range(output3D.shape[0]):
        
            fig = plt.figure(figsize=(9.6, 5.4))
            gs = gridspec.GridSpec(1, 1)
            gs.update(wspace=-0.00, hspace=0.05) 
            ax = plt.subplot(gs[0], projection='3d')
            post_out = output_3D[b, 0].cpu().detach().numpy()
            post_out = camera_to_world(post_out, R=rot, t=0)
            post_out[:, 2] -= np.min(post_out[:, 2])
            show3Dpose(post_out, ax, True)
            plt.savefig(output_dir_3D + f'{b}' + '_3D.png', dpi=200, format='png', bbox_inches = 'tight')
           
            
            fig = plt.figure(figsize=(9.6, 5.4))
            gs = gridspec.GridSpec(1, 1)
            gs.update(wspace=-0.00, hspace=0.05) 
            ax = plt.subplot(gs[0], projection='3d')
            pose_target = out_target[b, 0].cpu().detach().numpy()
            pose_target = camera_to_world(pose_target, R=rot, t=0)
            pose_target[:, 2] -= np.min(pose_target[:, 2])
            show3Dpose(pose_target, ax, True)
            plt.savefig(output_dir_3D + f'{b}' + '_3DGT.png', dpi=200, format='png', bbox_inches = 'tight')
        


        action_error = eval_cal.test_calculation(output_3D, out_target, action, action_error, args.dataset, subject)
    
    p1, p2, pck, auc = print_error(args.dataset, action_error, args.train)

    return p1, p2, pck, auc

if __name__ == '__main__':
    seed = args.seed

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.dataset == 'h36m':
        dataset_path = args.root_path + 'data_3d_' + args.dataset + '.npz'
        dataset = Human36mDataset(dataset_path, args)
        actions = define_actions(args.actions)

        if args.train:
            train_data = Fusion(args, dataset, args.root_path, train=True)
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                            shuffle=True, num_workers=int(args.workers), pin_memory=True)
        test_data = Fusion(args, dataset, args.root_path, train=False)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                            shuffle=False, num_workers=int(args.workers), pin_memory=True)
    elif args.dataset == '3dhp':
        dataset_path = args.root_path + 'data_3d_' + args.dataset + '.npz'
        dataset = Mpi_inf_3dhp_Dataset(dataset_path, args)
        actions = define_actions_3dhp(args.actions, 0)

        if args.train:
            train_data = Fusion_3dhp(args, dataset, args.root_path, train=True)
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                            shuffle=True, num_workers=int(args.workers), pin_memory=True)
        test_data = Fusion_3dhp(args, dataset, args.root_path, train=False)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                            shuffle=False, num_workers=int(args.workers), pin_memory=True)

    temporal_skeleton = list(range(0, args.frames))

    temporal_skeleton = np.array(temporal_skeleton)

    temporal_skeleton -= 1
    adj_temporal = adj_mx_from_skeleton_temporal(args.frames, temporal_skeleton)
    adj = adj_mx_from_skeleton(dataset.skeleton())
    model = Model(adj, adj_temporal, args).cuda()

    if args.previous_dir != '':
        Load_model(args, model)

    lr = args.lr
    all_param = []
    all_param += list(model.parameters())

    optimizer = optim.Adam(all_param, lr=lr, amsgrad=True)
    
    ##--------------------------------epoch-------------------------------- ##
    best_epoch = 0
    loss_epochs = []
    mpjpes = []

    for epoch in range(1, args.nepoch + 1):
        ## train
        if args.train: 
            loss = train(train_dataloader, model, optimizer, epoch)
            loss_epochs.append(loss * 1000)

        ## test
        with torch.no_grad():
            p1, p2, pck, auc = test(actions, test_dataloader, model)
            mpjpes.append(p1)
            
        if args.train:
            save_model_epoch(args.checkpoint, epoch, model)

        ## save the best model
        if args.train and p1 < args.previous_best:
            best_epoch = epoch
            args.previous_name = save_model(args, epoch, p1, model, 'model')
            args.previous_best = p1

        ## print
        if args.train:
            logging.info('epoch: %d, lr: %.6f, loss: %.4f, p1: %.2f, p2: %.2f' % (epoch, lr, loss, p1, p2))
            print('%d, lr: %.6f, loss: %.4f, p1: %.2f, p2: %.2f' % (epoch, lr, loss, p1, p2))
            
            ## adjust lr
            if epoch % args.lr_decay_epoch == 0:
                lr *= args.lr_decay_large
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay_large
            else:
                lr *= args.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay 
        else:
            if args.dataset == 'h36m':
                print('p1: %.2f, p2: %.2f' % (p1, p2))
            elif args.dataset == '3dhp':
                print('pck: %.2f, auc: %.2f, p1: %.2f, p2: %.2f' % (pck, auc, p1, p2))
            break

        ## training curves
        if epoch == 1:
            start_epoch = 3
                
        if args.train and epoch > start_epoch:
            plt.figure()
            epoch_x = np.arange(start_epoch+1, len(loss_epochs)+1)
            plt.plot(epoch_x, loss_epochs[start_epoch:], '.-', color='C0')
            plt.plot(epoch_x, mpjpes[start_epoch:], '.-', color='C1')
            plt.legend(['Loss', 'Test'])
            plt.ylabel('MPJPE')
            plt.xlabel('Epoch')
            plt.xlim((start_epoch+1, len(loss_epochs)+1))
            plt.savefig(os.path.join(args.checkpoint, 'loss.png'))
            plt.close()



