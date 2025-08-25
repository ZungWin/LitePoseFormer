import argparse
import os
import math
import time
import logging

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--spatial_embed_dim', default=32, type=int)
    parser.add_argument('--dataset', type=str, default='h36m')
    parser.add_argument('--keypoints', default='cpn_ft_h36m_dbb', type=str)
    parser.add_argument('--data_augmentation', type=int, default=1)
    parser.add_argument('--reverse_augmentation', type=bool, default=False)
    parser.add_argument('--test_augmentation', type=bool, default=True)
    parser.add_argument('--crop_uv', type=int, default=0)
    parser.add_argument('--root_path', type=str, default='dataset/')
    parser.add_argument('--actions', default='*', type=str)
    parser.add_argument('--downsample', default=1, type=int)
    parser.add_argument('--subset', default=1, type=float)
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--nepoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay_large', type=float, default=0.5)
    parser.add_argument('--lr_decay_epoch', type=int, default=10)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('-lrd', '--lr_decay', default=0.97, type=float)
    parser.add_argument('--frames', type=int, default=27)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--previous_dir', type=str, default='')
    parser.add_argument('--pos_embedding_type', type=str, default='sine-full')
    parser.add_argument('--num_keypoints', type=int, default=17)
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--drop_path_rate', type=float, default=0.)
    parser.add_argument('--attn_drop', type=float, default=0.0)
    parser.add_argument('--mlp_ratio', type=float, default=2.0)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--out_joints', type=int, default=17)
    parser.add_argument('--out_all', type=int, default=1)
    parser.add_argument('--previous_best', type=float, default= math.inf)
    parser.add_argument('--previous_name', type=str, default='')
    parser.add_argument('--apply_init', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=1234, help='random seed to use. Default=123')

    args = parser.parse_args()

    if args.test:
        args.train = 0

    args.pad = (args.frames-1) // 2
    stride_num = {
        '9': [3, 3, 3],
        '27': [3, 3, 3],
        '81': [9, 3, 3],
        '243': [3, 9, 9],
        '351': [3, 9, 13],
        }

    if str(args.frames) in stride_num:
        args.stride_num = stride_num[str(args.frames)]
    args.temporal_embed_dim = 128
    args.root_joint = 0
    if args.dataset == 'h36m':
        args.subjects_train = 'S1,S5,S6,S7,S8'
        args.subjects_test = 'S9,S11'

        args.n_joints = 17
        args.out_joints = 17

        args.joints_left = [4, 5, 6, 11, 12, 13] 
        args.joints_right = [1, 2, 3, 14, 15, 16]
            
    elif args.dataset.startswith('3dhp'):
        args.subjects_train = 'S1,S2,S3,S4,S5,S6,S7,S8' 
        args.subjects_test = 'TS1,TS2,TS3,TS4,TS5,TS6' # all
        # args.subjects_test = 'TS1,TS2' # GS
        # args.subjects_test = 'TS3,TS4' # no GS
        # args.subjects_test = 'TS5,TS6' # Outdoor

        args.n_joints, args.out_joints = 17, 17
        args.joints_left, args.joints_right = [4,5,6,11,12,13], [1,2,3,14,15,16]

    if args.train:
        logtime = time.strftime('%m%d_%H%M_%S_')

        args.checkpoint = 'checkpoint/' + logtime + '%d'%(args.frames)
        os.makedirs(args.checkpoint, exist_ok=True)

        args_write = dict((name, getattr(args, name)) for name in dir(args) if not name.startswith('_'))
        file_name = os.path.join(args.checkpoint, 'configs.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(args_write.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
            opt_file.write('==> Args:\n')

        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
            filename=os.path.join(args.checkpoint, 'train.log'), level=logging.INFO)

    return args




