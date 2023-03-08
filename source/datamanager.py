import os, random
import numpy as np
import tensorflow as tf
import source.utils as utils
from sklearn.utils import shuffle

class DataSet(object):

    def __init__(self, **kwargs):

        print("\nInitializing Dataset...")
        (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
        self.x_tr, self.y_tr = shuffle(x_tr, y_tr)
        self.x_te, self.y_te = x_te, y_te

        self.config = kwargs.copy()

        # select training set
        self.x_tr = self.x_tr[self.y_tr == self.config['select_norm']]
        self.y_tr = self.y_tr[self.y_tr == self.config['select_norm']]

        # split test set and validation set (test set 2)
        bound_val = int(self.x_te.shape[0] * 0.5)
        self.x_val, self.y_val = \
            self.x_te[bound_val:], self.y_te[bound_val:]
        self.x_te, self.y_te = \
            self.x_te[:bound_val], self.y_te[:bound_val]

        self.num_tr, self.num_te, self.num_val = \
            self.x_tr.shape[0], self.x_te.shape[0], self.x_val.shape[0]
        self.dim_h, self.dim_w, self.dim_c = \
            self.x_tr.shape[1], self.x_tr.shape[2], 1

        self.config['list_k'] = [4, 7]
        self.__reset_index()
        self.__make_samples4viz()

    def __reset_index(self):

        self.idx_tr, self.idx_te, self.idx_val = 0, 0, 0

    def __make_samples4viz(self):

        batch_name, batch_x, batch_m, batch_y = [], [], [], []
        for cls in range(10):
            tmp_x = np.expand_dims(utils.min_max_norm(self.x_te[self.y_te == cls][0]), axis=-1)
            tmp_m = utils.get_disjoint_mask(disjoint_n=self.config['disjoint_n'], dim_h=self.dim_h, dim_w=self.dim_w, list_k=self.config['list_k'])
            tmp_y = 1-int(cls==self.config['select_norm'])

            batch_name.append(cls)
            batch_x.append(tmp_x)
            batch_m.append(tmp_m)
            batch_y.append(tmp_y)

        batch_name.append(batch_name)
        batch_x = np.asarray(batch_x)
        batch_m = np.asarray(batch_m)
        batch_y = np.asarray(batch_y)

        self.batchviz = {'name':batch_name, 'x':batch_x.astype(np.float32), 'm':batch_m.astype(np.float32), 'y':batch_y.astype(np.float32)}

    def next_batch(self, batch_size=1, ttv=0):

        if(ttv == 0):
            idx_d, num_d, data, label = \
                self.idx_tr, self.num_tr, self.x_tr, self.y_tr
        elif(ttv == 1):
            idx_d, num_d, data, label = \
                self.idx_te, self.num_te, self.x_te, self.y_te
        else:
            idx_d, num_d, data, label = \
                self.idx_val, self.num_val, self.x_val, self.y_val

        batch_x, batch_m, batch_y, terminate = [], [], [], False
        while(True):

            try:
                tmp_x = np.expand_dims(utils.min_max_norm(data[idx_d]), axis=-1)
                tmp_m = utils.get_disjoint_mask(disjoint_n=self.config['disjoint_n'], dim_h=self.dim_h, dim_w=self.dim_w, list_k=self.config['list_k'])
                tmp_y = 1-int(label[idx_d]==self.config['select_norm']) # 0: normal, 1: abnormal
            except:
                idx_d = 0
                if(ttv == 0):
                    self.x_tr, self.y_tr = shuffle(self.x_tr, self.y_tr)
                terminate = True
                break

            batch_x.append(tmp_x)
            batch_m.append(tmp_m)
            batch_y.append(tmp_y)
            idx_d += 1

            if(len(batch_x) >= batch_size): break

        batch_x = np.asarray(batch_x)
        batch_m = np.asarray(batch_m)
        batch_y = np.asarray(batch_y)

        if(ttv == 0): self.idx_tr = idx_d
        elif(ttv == 1): self.idx_te = idx_d
        else: self.idx_val = idx_d

        return {'x':batch_x.astype(np.float32), 'm':batch_m.astype(np.float32), 'y':batch_y.astype(np.float32), 'terminate':terminate}
