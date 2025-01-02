import os
import argparse
import shelve
import logging
import numpy as np
import torch.multiprocessing as mp

from config import get_cfg_defaults  # local variable usage pattern
from src.util import prepare_dirs, set_seed, write_file, folder_name_generator, setup, cleanup
from src.training.train import train

def run_trials(rank: int, cfg: argparse.Namespace):
    '''
    Runs the train function for cfg.MISC.NUM_TRIALS times with identical input configurations (except for the random seed, which is altered).
    Args:
        rank: Identifying index of GPU running process
        cfg: Input configurations
    Returns:
        None
    '''
    local_rank = rank
    # set up configurations
    directory = cfg.MISC.DIR
    molecule_name = cfg.DATA.MOLECULE
    num_trials = cfg.MISC.NUM_TRIALS
    random_seed = cfg.MISC.RANDOM_SEED
    result_logger_name = cfg.EVAL.RESULT_LOGGER_NAME
    node_idx = cfg.DDP.NODE_IDX
    local_world_size = cfg.DDP.LOCAL_WORLD_SIZE
    world_size = cfg.DDP.WORLD_SIZE
    global_rank = node_idx * local_world_size + local_rank
    master_addr = cfg.DDP.MASTER_ADDR
    master_port = cfg.DDP.MASTER_PORT
    use_same_local_seed = cfg.MISC.SAME_LOCAL_SEED
    logging.info(f"Running DDP on rank {global_rank}.")
    if world_size > 1:
        setup(global_rank, world_size, master_addr, master_port)
    if global_rank == 0:
        prepare_dirs(cfg)
    best_score = float('inf')*np.ones(3)
    avg_time_elapsed = 0.0
    result_dic = {}
    write_file(result_logger_name, f"=============== {directory.split('/')[-1]} ===============", global_rank)
    current_model_path = os.path.join(cfg.MISC.SAVE_DIR, 'last_model.pth')
    best_model_path = os.path.join(cfg.MISC.SAVE_DIR, 'best_model_pth')
    for trial in range(num_trials):
        seed = random_seed + trial * 1000
        # set random seeds
        set_seed(seed + (0 if use_same_local_seed else global_rank))
        new_score, time_elapsed, dic = train(cfg, local_rank, global_rank)
        new_score = np.array(new_score)
        result_log = f"[{molecule_name}] Score {new_score}, Time elapsed {time_elapsed:.4f}"
        write_file(result_logger_name, f"Trial - {trial+1}", global_rank)
        write_file(result_logger_name, result_log, global_rank)
        if new_score[0] < best_score[0]:
            best_score = new_score
            if global_rank == 0:
                os.system(f'mv "{current_model_path}" "{best_model_path}"')
        avg_time_elapsed += time_elapsed / num_trials
        if dic is not None:
            for key in dic:
                if key in result_dic:
                    result_dic[key] = np.concatenate((result_dic[key], np.expand_dims(dic[key], axis=0)), axis=0)
                else:
                    result_dic[key] = np.expand_dims(dic[key], axis=0)
    result_log = f"[{directory.split('/')[-1]}][{molecule_name}] Best Score {best_score}, Time elapsed {avg_time_elapsed:.4f}, over {num_trials} trials"
    write_file(result_logger_name, result_log, global_rank)
    if global_rank == 0:
        np.save(os.path.join(directory, 'result.npy'), result_dic)
    if world_size > 1:
        cleanup()


def main(cfg: argparse.Namespace):
    '''
    Main function for NQS program
    Args:
        cfg: input configurations
    Returns:
        None
    '''
    world_size = cfg.DDP.WORLD_SIZE
    local_world_size = cfg.DDP.LOCAL_WORLD_SIZE
    if world_size > 1:
        mp.spawn(run_trials, args=(cfg,), nprocs=local_world_size, join=True)
    else:
        run_trials(0, cfg)
    logging.info('--------------- Finished ---------------')


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(description="Command-Line Options")
    parser.add_argument(
        "--config_file",
        default="",
        metavar="FILE",
        help="Path to the yaml config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        '--list_molecules',
        action='store_true',
        default=False,
        help='List saved molecules (instead of running program)',
    )
    args = parser.parse_args()
    # Remainder of program does not run if program is asked to list molecules
    if args.list_molecules:
        molecule_list = 'PubChem IDs stored by program:'
        with shelve.open('./src/data/pubchem_IDs/ID_data') as pubchem_ids:
            for name, id in pubchem_ids.items():
                molecule_list+=('\n{}: {}'.format(name, id))
        print(molecule_list+'\nStored names can be used as input arguments to access corresponding IDs.')
    else:
        # configurations
        cfg = get_cfg_defaults()
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        # set up directories (cfg.MISC.DIR)
        log_folder_name = folder_name_generator(cfg, args.opts)
        if cfg.MISC.DIR == '':
            cfg.MISC.DIR = './logger/{}'.format(log_folder_name)
        if cfg.MISC.SAVE_DIR == '':
            cfg.MISC.SAVE_DIR = './logger/{}'.format(log_folder_name)
        os.makedirs(cfg.MISC.DIR, exist_ok=True)
        os.makedirs(cfg.MISC.SAVE_DIR, exist_ok=True)
        os.system(f'cp "{args.config_file}" "{cfg.MISC.DIR}"')
        # freeze the configurations
        cfg.freeze()
        # set logger
        logging.root.handlers = []
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(cfg.MISC.DIR, 'logging.log')),
                logging.StreamHandler()
            ]
        )
        # run program
        main(cfg)
    
