"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
import cv2
import matplotlib.pyplot as plt

import os
from os.path import join as ospj
import json
from shutil import copyfile

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils


# def save_json(json_file, filename):
#     with open(filename, 'w') as f:
#         json.dump(json_file, f, indent=4, sort_keys=False)


# def print_network(network, name):
#     num_params = 0
#     for p in network.parameters():
#         num_params += p.numel()
#     # print(network)
#     print("Number of parameters of %s: %i" % (name, num_params))


# def he_init(module):
#     if isinstance(module, nn.Conv2d):
#         nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
#         if module.bias is not None:
#             nn.init.constant_(module.bias, 0)
#     if isinstance(module, nn.Linear):
#         nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
#         if module.bias is not None:
#             nn.init.constant_(module.bias, 0)


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x, ncol, filename):
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


@torch.no_grad()
def translate_using_reference(nets, args, x_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.size()
    wb = torch.ones(1, C, H, W).to(x_src.device)
    x_src_with_wb = torch.cat([wb, x_src], dim=0)

    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    s_ref = nets.style_encoder(x_ref, y_ref)
    s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
    x_concat = [x_src_with_wb]
    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        x_fake_with_ref = torch.cat([x_ref[i:i+1], x_fake], dim=0)
        x_concat += [x_fake_with_ref]

    img__ = tr_image(x_fake)
    plt.imsave(args.result_dir+'/out.png', img__)
    
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N+1, filename)
    del x_concat


# 이미지 변환 (tensor to np)
def tr_image(img):
    img_ = img.squeeze()
    img_ = img_.detach().cpu().numpy()
    img_ = np.transpose(img_, (1, 2, 0))
    mean_, std_ = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])
    img_ = np.clip(255.0 * (img_ * std_ + mean_), 0, 255)
    img_ = img_.astype(np.uint8)
    return img_