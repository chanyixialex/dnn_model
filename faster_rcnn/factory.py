import numpy as py
from faster_rcnn.rw_car import rw_car
'''
gets imdbs by name

'''
__sets = {}
def _selective_datasets(split):
    imdb = rw_car(split)
    return imdb
for split in ['train', 'test']:
    name = '{}'.format(split)
    __sets[name] = rw_car(name)

# load datasets
def get_imdb(name):
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset:{}'.format(name))
    return __sets[name]()
