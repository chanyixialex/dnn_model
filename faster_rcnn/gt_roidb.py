import numpy as np
from faster_rcnn.config import cfg
import PIL
import os
import pickle

def prepare_roidb(imdb):
    cache_file = os.path.join(imdb.cache_path, imdb.name + '_gt_roidb.pkl')
def add_bbox_regression_targets(roidb):
    print()
