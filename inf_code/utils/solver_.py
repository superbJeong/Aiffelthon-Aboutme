"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from os.path import join as ospj
from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_ import build_model
from utils.checkpoint_ import CheckpointIO
from utils.data_loader_ import InputFetcher
import utils.utils_ as utils



class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cpu')


        self.nets_ema = build_model(args)
        # print(self.nets_ema["generator"])
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '050000_nets_ema.ckpt'), data_parallel=False, **self.nets_ema)]

        self.to(self.device)
        # for name, network in self.named_children():
        #     # Do not initialize the FAN parameters
        #     if ('ema' not in name) and ('fan' not in name):
        #         print('Initializing %s...' % name)
        #         network.apply(utils.he_init)

    # def _save_checkpoint(self):
    #     for ckptio in self.ckptios:
    #         ckptio.save()

    def _load_checkpoint(self):
        for ckptio in self.ckptios:
            ckptio.load()

    # def _reset_grad(self):
    #     for optim in self.optims.values():
    #         optim.zero_grad()


    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint()

        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))
        ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))

        fname = ospj(args.result_dir, 'reference.jpg')
        print('Working on {}...'.format(fname))
        utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)

