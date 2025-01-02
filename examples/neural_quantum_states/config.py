from yacs.config import CfgNode as CN
# yacs official github page https://github.com/rbgirshick/yacs

_C = CN()
''' Miscellaneous '''
_C.MISC = CN()
# Random seed
_C.MISC.RANDOM_SEED = 0
# Logger path
_C.MISC.DIR = ''
# Saved models path
_C.MISC.SAVE_DIR = ''
# Number of trials
_C.MISC.NUM_TRIALS = 0
# DDP seeds
_C.MISC.SAME_LOCAL_SEED = False

''' Training hyper-parameters '''
_C.TRAIN = CN()
# Learning rate
_C.TRAIN.LEARNING_RATE = 0.0
# Optimizer weight decay
_C.TRAIN.WEIGHT_DECAY = 0.0
# Training optimizer name
# Available choices: ['adam', 'sgd', 'adadelta', 'adamax', 'adagrad', 'nadam', 'radam', 'adamw']
_C.TRAIN.OPTIMIZER_NAME = ''
# Learning rate scheduler name
# Available choices: ['decay', 'cyclic', 'trap', 'const', 'cosine', 'cosine_warm']
_C.TRAIN.SCHEDULER_NAME = ''
# Training batch size
_C.TRAIN.BATCH_SIZE = 0
# Number of training epochs
_C.TRAIN.NUM_EPOCHS = 0
# How many iterations between recalculation of Hamiltonian energy
_C.TRAIN.RETEST_INTERVAL = 1
# Whether to use entropy regularization
_C.TRAIN.ANNEALING_COEFFICIENT = 0.0
# Entropy regularization constant schedule type
_C.TRAIN.ANNEALING_SCHEDULER = 'none'
# Largest number of samples that each GPU is expected to process at one time
_C.TRAIN.MINIBATCH_SIZE = 1

''' Model hyper-parameters '''
_C.MODEL = CN()
# The name of the model
# Available choices: ['made', 'transformer', 'retnet']
_C.MODEL.MODEL_NAME = ''
# The number of hidden layers in MADE/Phase MLP
_C.MODEL.MADE_DEPTH = 1
# Hidden layer size in MADE/Phase MLP
_C.MODEL.MADE_WIDTH = 64
# Embedding/Hidden State Dimension for Transformer/RetNet
_C.MODEL.EMBEDDING_DIM = 32
# Number of Attention/Retention Heads per Transformer/RetNet layer
_C.MODEL.ATTENTION_HEADS = 4
# Feedforward Dimension for Transformer/RetNet
_C.MODEL.FEEDFORWARD_DIM = 512
# Number of Transformer/RetNet layers
_C.MODEL.TRANSFORMER_LAYERS = 1
# Parameter std initialization
_C.MODEL.INIT_STD = 0.1
# Model temperature parameter
_C.MODEL.TEMPERATURE = 1.0

''' Data hyper-parameters '''
_C.DATA = CN()
# Minimum nonunique batch size for autoregressive sampling
_C.DATA.MIN_NUM_SAMPLES = 1e2
# Maximum nonunique batch size for autoregressive sampling
_C.DATA.MAX_NUM_SAMPLES = 1e12
# Molecule name
_C.DATA.MOLECULE = ''
# Pubchem molecular compound CID (only needed if name does not exist in shelf)
_C.DATA.MOLECULE_CID = 0
# Basis ['STO-3G', '3-21G', '6-31G', '6-311G*', '6-311+G*', '6-311++G**', '6-311++G(2df,2pd)', '6-311++G(3df,3pd)', 'cc-pVDZ', 'cc-pVDZ-DK', 'cc-pVTZ', 'cc-pVQZ', 'aug-cc-pCVQZ']
_C.DATA.BASIS = ''
# Prepare FCI result; not recommended for medium-to-large systems
_C.DATA.FCI = False
# Choice of Hamiltonian to use for training: ['adaptive_shadows', 'exact', 'sample']
_C.DATA.HAMILTONIAN_CHOICE = 'exact'
# Sample batch size for Hamiltonian sampling
_C.DATA.HAMILTONIAN_BATCH_SIZE = 10
# Number of unique samples in Hamiltonian sampling
_C.DATA.HAMILTONIAN_NUM_UNIQS = 500
# Probability of resampling estimated Hamiltonian
_C.DATA.HAMILTONIAN_RESET_PROB = 0.01
# Number of batches that the flipped input states are split into during local energy calculation
_C.DATA.HAMILTONIAN_FLIP_BATCHES = 1

''' Evaluation hyper-parameters '''
_C.EVAL = CN()
# Loading path of the saved model
_C.EVAL.MODEL_LOAD_PATH = ''
# Name of the results logger
_C.EVAL.RESULT_LOGGER_NAME = './results/results.txt'

''' DistributedDataParallel '''
_C.DDP = CN()
# Node number globally
_C.DDP.NODE_IDX = 0
# This needs to be explicitly passed in
_C.DDP.LOCAL_WORLD_SIZE = 1
# Total number of GPUs
_C.DDP.WORLD_SIZE = 1
# Master address for communication
_C.DDP.MASTER_ADDR = 'localhost'
# Master port for communication
_C.DDP.MASTER_PORT = 12355


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for the project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`
