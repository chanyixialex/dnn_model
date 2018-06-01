from faster_rcnn.imdb import imdb
import csv
import os
import _pickle as pkl
import numpy as np
import scipy.sparse
class rw_car(imdb):
    def __init__(self, image_set, data_path=None):  #image_set:dataset type,e.g. train,trainval,or test datasets.
        imdb.__init__(self, 'car_' + image_set)
        self._image_set = image_set  # train or test
        self._data_path = data_path  # datasets path '../data/car'
        self._classes = ('__background__', 'car')  # object type in image,car and background
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes))) #construct dict:{'__background__':0, 'car': 1}
        self._image_ext = '.jpg'  # suffix of image
        self._image_name = self._load_image_set_name() #read image's name from 'train.csv'
        #self._roidb_handler = self.gt_roidb # get gt of image
        self._roidb_handler = self.gt_roidb  # 获取图片的gt
        self.config = {'cleanup' : True,
                       'user_salt' : True,
                       'top_k' : 2000} # specific config options
        #assert os.path.exists(self._data_path), 'Image path does not exist: {}'.format(self._data_path)
        #exist,if image path does not exist

    #get image_set name
    def _load_image_set_name(self):
        image_set_file = os.path.join(self._data_path, self._image_set + '_1w.csv')
        # image_set_file is ../data/car/train_1w.csv
        assert os.path.exists(image_set_file), 'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file, 'r') as f:
            image_name = []
            rows = csv.reader(f)
            n_row = 0
            for row in rows:
                if n_row != 0:
                    image_name.append(row[0].strip())
                n_row = n_row + 1
        return image_name
    #get image's path by image's name
    def _image_path_from_name(self, name):
        image_path = os.path.join(self._data_path, self._image_set, self._image_name)
        assert os.path.exists(image_path), 'image path does not exists:{}'.format(image_path)

    def gt_roidb(self):
        '''
        import gt of image to .pkl file in order to speed up
        :return: the database of ground-truth regions of interest.

        '''
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as field:
                roidb = pkl.load(field)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb
        gt_roidb = self._load_annotation()  # read and handle all gt file
        with open(cache_file, 'wb') as field:
            pkl.dump(gt_roidb, field, -1)
        print('write gt roidb to {}'.format(cache_file))
        return gt_roidb
    def _load_annotation(self):
        gt_roidb = []
        csvfile = os.path.join(self._data_path, self._image_set + '_1w.csv')
        f = open(csvfile)
        rows = csv.reader(f)
        n_row = 0
        for row in rows:
            if n_row != 0 and n_row > 2851:
                row_boxes = row[1].strip().split(';')
                num_objs = len(row_boxes)
                boxes = np.zeros((num_objs, 4), dtype=np.uint16)
                gt_classes = np.zeros((num_objs), dtype=np.int32)#对象数量
                overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)#某个对象对应哪个类别
                for i in range(num_objs):
                    #row_boxes_indexs = np.zeros((4), dtype=np.float32)
                    row_boxes_indexs = row_boxes[i].strip().split('_')
                    row_boxes_indexs = list(map(float, row_boxes_indexs))
                    #row_boxes_indexs = np.array(row_boxes_indexss)
                    print()
                    x1 = row_boxes_indexs[0]
                    y1 = row_boxes_indexs[1]
                    x2 = row_boxes_indexs[0] + row_boxes_indexs[2]
                    y2 = row_boxes_indexs[1] + row_boxes_indexs[3]
                    boxes[i, :] = [x1, y1, x2, y2]
                    gt_classes[i] = '1'  # 1:indicates car ;0: indicates background
                    overlaps[i, 1] = 1.0
                overlaps = scipy.sparse.csr_matrix(overlaps)
                gt_roidb.append({'boxes': boxes, 'gt_classes': gt_classes, 'gt_overlaps': overlaps, 'flipped': False})
            n_row = n_row + 1
            print(n_row)

        f.close()
        print(n_row)
        return gt_roidb


if __name__ == '__main__':
    #a = ['16.620689999999968', '29.620689999999996']
    #a = list(map(float,a))
    #print(a)
    d = rw_car('train', '../data/car')
    print(d.gt_roidb())
    print(len(d.gt_roidb()['boxes']))



