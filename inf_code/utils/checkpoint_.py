"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import torch


class CheckpointIO(object):
    def __init__(self, fname_template, data_parallel=False, **kwargs):
        os.makedirs(os.path.dirname(fname_template), exist_ok=True)
        self.fname_template = fname_template
        self.module_dict = kwargs
        self.data_parallel = data_parallel

    def register(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self):
        # fname = self.fname_template.format(step)
        # print('Saving checkpoint into %s...' % fname)
        outdict = {}
        # for name, module in self.module_dict.items():
        #     print(type(module))
        #     outdict[name] = module.state_dict()
        # for name, module in self.module_dict.items():
        #     print(module)
        #     outdict[name] = module.state_dict()
        # print(self.module_dict)

        # torch.save(outdict, fname)
        # torch.save(outdict, "state_dict.pt")
        # torch.save(self.module_dict, "entire.pt")

    def load(self, step):
        fname = self.fname_template.format(step)
        assert os.path.exists(fname), fname + ' does not exist!'
        print('Loading checkpoint from %s...' % fname)
        
        module_dict = torch.load(fname, map_location=torch.device('cpu'))

        for name, module in self.module_dict.items():
            module.load_state_dict(module_dict[name])

            torch.save(module.state_dict(), f"{name}_test.pt")
