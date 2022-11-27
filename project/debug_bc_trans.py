import argparse

import robomimic
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.macros as Macros
from robomimic.config.base_config import BaseConfig
from robomimic.scripts.train import train
import os
import json

robomimic_path = robomimic.__path__[0]

# Dataset path
# dataset_path = os.path.join(robomimic_path,'../datasets/lift/ph/images.hdf5') 
# out_path = os.path.join(robomimic_path,'../project/out')
# print(os.path.abspath(dataset_path))
# print(os.path.abspath(out_path))
# print(os.getcwd())

debug = False
# Turn debug mode on possibly
if debug:
    Macros.DEBUG = True

# load config
config_path = os.path.join(robomimic_path,'exps/templates/bc_trans_images.json')
config = BaseConfig(json.load(open(config_path)))

# set torch device
device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

# run training
train(config, device=device)