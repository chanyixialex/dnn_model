import os.path as osp
import os
from faster_rcnn.config import cfg
class imdb(object):
    def __init__(self, name):
        self._name = name
        self._classes = []
        self._num_classes = 0
        self._roidb = None  # gets roidb
        self._roidb_handler = self.default_roidb  #ways of getting roidb

    def default_roidb(self):
        raise NotImplementedError
    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    @property
    def name(self):
        return self._name

    @property
    def roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()
        return self._roidb

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path