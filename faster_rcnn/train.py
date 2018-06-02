import tensorflow as tf
import numpy as np
import os
from faster_rcnn.config import cfg

'''
use minibatch for datasets
fast r_cnn
'''
class SolverWrapper(object):

    def __init__(self, sess, saver, network, imdb, roidb, output_dir, pretrained_model=None):
        self.network = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

def get_training_roidb(imdb):
    print()