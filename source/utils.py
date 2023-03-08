import os, glob, shutil, json, pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import ndimage
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc

def make_dir(path, refresh=False):

    try: os.mkdir(path)
    except:
        if(refresh):
            shutil.rmtree(path)
            os.mkdir(path)

def sorted_list(path):

    tmplist = glob.glob(path)
    tmplist.sort()

    return tmplist

def key_parsing(dic, key, default):

    try: return dic[key]
    except: return default

def read_json(path):

    with open(path, "r") as json_file:
        dic = json.load(json_file)

    return dic

def save_json(path, dic):

    list_del = []
    for idx_key, name_key in enumerate(dic.keys()):
        value = dic[name_key]
        if(isinstance(value, int) or isinstance(value, float)): pass
        elif(isinstance(value, np.int64)): dic[name_key] = int(value)
        elif(isinstance(value, np.float32)): dic[name_key] = float(value)
        else: dic[name_key] = str(value)

    with open(path, 'w') as json_file:
        json.dump(dic, json_file)

def save_pkl(path, pkl):

    with open(path,'wb') as fw:
        pickle.dump(pkl, fw)

def load_pkl(path):

    with open(path, 'rb') as fr:
        pkl = pickle.load(fr)

    return pkl

def min_max_norm(x):

    return (x - x.min()) / (x.max() - x.min() + (1e-31))

def get_disjoint_mask(disjoint_n, dim_h, dim_w, list_k=[2, 4]):

    masks = []
    for cell_k in list_k:
        cell_num = (dim_h//cell_k * dim_w//cell_k)
        indicies = np.linspace(0, cell_num, cell_num+1)[:-1]
        indicies.shape
        indicies = shuffle(indicies)

        bound = indicies.shape[0] // disjoint_n
        for idx_d in range(disjoint_n):
            mask_i = np.ones((cell_num))
            split_s = bound*idx_d
            split_e = bound*(idx_d+1)
            if(idx_d >= disjoint_n-1):
                disjoint_i = indicies[split_s:]
            else:
                disjoint_i = indicies[split_s:split_e]
            for d_i in disjoint_i:
                mask_i[int(d_i)] = 0
            mask_i = mask_i.reshape(dim_h//cell_k, dim_w//cell_k)
            mask_i = ndimage.zoom(mask_i, zoom=cell_k, order=0)
            masks.append(mask_i)

    masks = np.asarray(masks)
    return masks

def masking(img, mask, inverse=False):

    images, maskes = [], []
    for idx_mask in range(mask.shape[0]):
        images.append(img)
        maskes.append(img * np.expand_dims(mask[idx_mask], axis=-1))

    images = np.asarray(images)
    maskes = np.asarray(maskes)

    return images, maskes

def preparing_disjoint_mask(disjoint_n, dim_h, dim_w, num_preset=50, savedir='disjoint_preset'):

    make_dir(path=savedir, refresh=True)
    for idx_set in range(num_preset):
        masks = []
        for cell_k in [2, 4]:
            cell_num = (dim_h//cell_k * dim_w//cell_k)
            indicies = np.linspace(0, cell_num, cell_num+1)[:-1]
            indicies.shape
            indicies = shuffle(indicies)

            bound = indicies.shape[0] // disjoint_n
            for idx_d in range(disjoint_n):
                mask_i = np.ones((cell_num))
                split_s = bound*idx_d
                split_e = bound*(idx_d+1)
                if(idx_d >= disjoint_n-1):
                    disjoint_i = indicies[split_s:]
                else:
                    disjoint_i = indicies[split_s:split_e]
                for d_i in disjoint_i:
                    mask_i[int(d_i)] = 0
                mask_i = mask_i.reshape(dim_h//cell_k, dim_w//cell_k)
                mask_i = ndimage.zoom(mask_i, zoom=cell_k, order=0)
                masks.append(mask_i)

        masks = np.asarray(masks)
        np.save(os.path.join(savedir, '%05d.npy' %(idx_set)), masks)

def measure_auroc(labels, scores, savepath=None):

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    auroc = auc(fpr, tpr)

    dic_score = {'Label':labels, 'Loss':scores}
    df_socre = pd.DataFrame.from_dict(dic_score)
    df_socre.loc[df_socre['Label'] == 0, 'Label'] = 'Good'
    df_socre.loc[df_socre['Label'] == 1, 'Label'] = 'Not Good'

    if(not(savepath is None)):
        plt.figure(figsize=(4, 3), dpi=100)
        ax1 = plt.subplot(1,1,1)

        ax1.set_title('AUROC: %.5f' %(auroc))
        sns.violinplot(x="Label", y="Loss", data=df_socre)

        plt.tight_layout()
        plt.savefig(savepath, transparent=True)
        plt.close()

    return auroc

def plot_generation(y, y_hat, map, savepath=""):

    gray = y.shape[-1] == 1
    len_plot = y.shape[0]

    plt.figure(figsize=(1.5*len_plot, 1.5*4), dpi=100)

    for idx_y in range(y.shape[0]):
        plt.subplot(4, len_plot, idx_y+1)
        if(gray): plt.imshow(y[idx_y, :, :, 0],  cmap='gray')
        else: plt.imshow(y[idx_y])
        plt.xticks([])
        plt.yticks([])
        if(idx_y == 0):
            plt.ylabel("$I$")

        plt.subplot(4, len_plot, idx_y+1+len_plot)
        if(gray): plt.imshow(y_hat[idx_y, :, :, 0],  cmap='gray')
        else: plt.imshow(y_hat[idx_y])
        plt.xticks([])
        plt.yticks([])
        if(idx_y == 0):
            plt.ylabel("$I_{r}$")

        plt.subplot(4, len_plot, idx_y+1+(len_plot*2))
        plt.imshow(map[idx_y], cmap='jet')
        plt.xticks([])
        plt.yticks([])
        if(idx_y == 0):
            plt.ylabel("$G(I, I_{r})$")

        plt.subplot(4, len_plot, idx_y+1+(len_plot*3))
        if(gray): plt.imshow(y[idx_y, :, :, 0],  cmap='gray')
        else: plt.imshow(y[idx_y])
        plt.imshow(map[idx_y], cmap='jet', alpha=0.5)
        plt.xticks([])
        plt.yticks([])
        if(idx_y == 0):
            plt.ylabel("Overlay")

    plt.tight_layout()
    plt.savefig(savepath, transparent=True)
    plt.close()

def nhwc2nchw(x):

    return np.transpose(x, [0, 3, 1, 2])

def nchw2nhwc(x):

    return np.transpose(x, [0, 2, 3, 1])

def detach(x):

    try: x = x.detach().numpy()
    except:
        try:
            x = x.cpu().detach().numpy()
        except:
            pass

    return x
