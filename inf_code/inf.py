"""

About Me
https://github.com/junnnn-a/About_Me

[220602]
usr 폴더에 있는 사용자 이미지를 crop하고 align하여 src 폴더에 저장하고,
이를 source 이미지로 하여 inference 하는 코드로 변경

[220531]
* Setting environment
conda create -n stargan-v2 python=3.6.7
conda activate stargan-v2
conda install -y pytorch torchvision cpuonly -c pytorch
pip install opencv-python==4.1.2.30 scikit-image==0.16.2 munch==2.5.0
conda install -c conda-forge dlib

"""

"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""



import os
import argparse
import sys

from munch import Munch
from torch.backends import cudnn
import torch

from utils.data_loader_ import get_test_loader
from utils.solver_ import Solver
from utils.crop_ import crop_align
import utils.scores_ as scr


def str2bool(v):
    return v.lower() in ('true')


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]


def main(args):
    # print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    # lm = crop_align()
    # usr_score = scr.get_scores_list(lm)
    # print(usr_score)
    # # sys.exit()
    # final_scores = scr.calulate_scores(usr_score)
    # # ref_name = scr.select_ref(final_scores)
    # # print(ref_name)
    # refs = scr.select_refs(final_scores)
    # print(refs)

    # sys.exit()

    solver = Solver(args)

    if args.mode == 'sample':
        assert len(subdirs(args.src_dir)) == args.num_domains
        assert len(subdirs(args.ref_dir)) == args.num_domains
        loaders = Munch(src=get_test_loader(root=args.src_dir,
                                            img_size=args.img_size,
                                            # batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers),
                        ref=get_test_loader(root=args.ref_dir,
                                            img_size=args.img_size,
                                            # batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers))
        solver.sample(loaders)
    elif args.mode == 'align':
        from utils.wing_ import align_faces
        align_faces(args, args.inp_dir, args.out_dir)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=1,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')

    parser.add_argument('--w_hpf', type=float, default=1,
                        help='weight for high-pass filtering')


    # misc
    parser.add_argument('--mode', type=str, default='sample',
                        choices=['sample', 'align'],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')

    # directory for training
    parser.add_argument('--checkpoint_dir', type=str, default='models',
                        help='Directory for saving network checkpoints')

    # directory for testing
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory for saving generated images')
    parser.add_argument('--src_dir', type=str, default='inputs/src',
                        help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default='inputs/ref',
                        help='Directory containing input reference images')
    parser.add_argument('--inp_dir', type=str, default='inputs/usr',
                        help='input directory when aligning faces')
    parser.add_argument('--out_dir', type=str, default='inputs/src/female',
                        help='output directory when aligning faces')

    # face alignment
    parser.add_argument('--wing_path', type=str, default='models/wing.ckpt')
    parser.add_argument('--lm_path', type=str, default='models/celeba_lm_mean.npz')

    args = parser.parse_args()

    main(args)