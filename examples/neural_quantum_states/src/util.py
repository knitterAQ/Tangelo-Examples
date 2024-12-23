import os
import random
import logging
import numpy as np
import torch
import torch.optim as optim
import torch.cuda as cuda
import torch.distributed as dist
import argparse
from typing import Any

def setup(global_rank: int, world_size: int, master_addr: str, master_port: int):
    # Necessary setup steps to initialize DDP process group
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    # initialize the process group
    cuda.set_device(global_rank)
    dist.init_process_group("nccl", rank=global_rank, world_size=world_size)

def cleanup(): # Cleans up DDP process group
    dist.destroy_process_group()

def set_seed(seed: int):
    # Sets random seeds for random, numpy, and torc
    # Required for reproducibility and for accurate parallelization on multi-GPU setups, see https://pytorch.org/docs/master/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def folder_name_generator(cfg: argparse.Namespace, opts): # Generates names of logging folders based on config arguments
    name_str = []
    for i,arg in enumerate(opts):
        if i % 2 == 1:
            name_str.append('{}'.format(arg))
    return '-'.join(name_str)

def prepare_dirs(cfg: argparse.Namespace): # Prepares appropriate directories needed for training/logging
    if not os.path.exists('./logger'):
        os.makedirs('./logger')
    if not os.path.exists(cfg.MISC.DIR):
        os.makedirs(cfg.MISC.DIR)
    # set logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(cfg.MISC.DIR, 'debug.log')),
            logging.StreamHandler()
        ]
    )

def write_file(file_name: str, content: Any, local_rank: int=0, overwrite: bool=False):
    # Writes 'content' to file at 'file_name'
    if local_rank == 0:
        if overwrite:
            f=open(file_name, "w+")
        else:
            f=open(file_name, "a+")
        f.write(content)
        f.write("\n")
        f.close()

def get_optimizer(opt_name: str, model: torch.nn.Module, learning_rate: float, weight_decay: float=0.0) -> optim.Optimizer:
    '''
    Retrieves an optimizer based on user selection
    Args:
        opt_name: name of optimizer choice
        model: NQS ansatz
        learning_rate: initial learning rate
        weight_decay: weight decay parameter for optimizer
    Returns:
        optim.Optimizer: PyTorch Optimizer object
    '''
    if opt_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    elif opt_name == 'adadelta':
        optimizer = optim.Adadelta(model.parameters())
    elif opt_name == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
    elif opt_name == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    elif opt_name == 'lbfgs':
        optimizer = optim.LBFGS(model.parameters(), lr=learning_rate)
    elif opt_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif opt_name == 'nadam':
        optimizer = optim.NAdam(model.parameters(), lr=learning_rate, weight_decay = weight_decay, decoupled_weight_decay=True)
    elif opt_name == 'radam':
        optimizer = optim.RAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)# decoupled_weight_decay=True)
    else:
        raise "opt_name not recognized."
    return optimizer
