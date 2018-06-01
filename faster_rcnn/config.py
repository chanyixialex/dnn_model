from easydict import EasyDict as edict
import os.path as osp
# Use RPN to detect objects
__C = edict()
cfg = __C
#__C.TRAIN.HAS_RPN = False
__C.IS_MULTISCALE = False
# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))