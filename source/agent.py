import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import source.utils as utils
import source.losses as lossf
import source.connector as con

import torchsummary
from torch.utils.tensorboard import SummaryWriter

class Agent(object):

    def __init__(self, **kwargs):

        print("\nInitializing Neural Network...")
        self.config = kwargs.copy()

        self.model = con.connect(nn=self.config['nn'])

        self.config['filters'] = [self.config['dim_c'], 64, 128, 256, 512]

        self.msgms = lossf.MSGMSLoss(num_scales=3, in_channels=self.config['dim_c'])

        self.__model = self.model.Neuralnet( \
            dim_h=self.config['dim_h'], dim_w=self.config['dim_w'], dim_c=self.config['dim_c'], \
            ksize=self.config['ksize'], filters=self.config['filters'], \
            ngpu=self.config['ngpu'], device=self.config['device']).to(self.config['device'])
        # if((self.config['device'].type == 'cuda') and (self.config['ngpu'] > 0)):
        #     self.__model = nn.DataParallel(self.__model, list(range(self.config['ngpu'])))

        self.__init_propagation(path=self.config['path_ckpt'])

    def __init_propagation(self, path):

        utils.make_dir(self.config['path_ckpt'], refresh=False)
        self.save_params()

        out = torchsummary.summary(\
            self.__model, [\
                (self.config['dim_c'], self.config['dim_h'], self.config['dim_w']), \
                (self.config['dim_c'], self.config['dim_h'], self.config['dim_w'])])

        if(self.config['mode_optim'].lower() == 'sgd'):
            self.optimizer = optim.SGD(self.__model.parameters(), lr=self.config['learning_rate'], momentum=0.1)
        elif((self.config['mode_optim'].lower() == 'rms') or self.config['mode_optim'].lower() == 'rmsprop'):
            self.optimizer = optim.RMSprop(self.__model.parameters(), lr=self.config['learning_rate'])
        elif(self.config['mode_optim'].lower() == 'adam'):
            self.optimizer = optim.Adam(self.__model.parameters(), lr=self.config['learning_rate'])
        else:
            self.optimizer = optim.SGD(self.__model.parameters(), lr=self.config['learning_rate'], momentum=0.9)

        self.lr_max, self.lr_min, self.lr_tmp = self.config['learning_rate'], self.config['learning_rate'] / 5, 0
        self.T_i, self.T_mult, self.T_cur = 20, 2, 0

        self.writer = SummaryWriter(log_dir=self.config['path_ckpt'])

    def step(self, minibatch, iteration=0, epoch=0, iter_per_epoch=-1, training=False):

        x, m = minibatch['x'], minibatch['m']
        num_batch, num_mask = x.shape[0], m.shape[1]
        x_new, y_new = [], []
        for idx_x in range(num_batch):
            images, maskes = utils.masking(x[idx_x].copy(), m[idx_x].copy(), inverse=False)
            x_new.extend(maskes)
            y_new.extend(images)
        x_new = np.asarray(x_new).astype(np.float32)
        y_new = np.asarray(y_new).astype(np.float32)
        x_new, y_new

        losses = {'msgms_b':[], 'msgms':[], \
            'ssim_b':[], 'ssim':[], \
            'l2_b':[], 'l2':[], \
            'opt_b':[], 'opt':[]}
        y_tot, y_hat_tot, x_tot = None, None, None
        for idx_iter in range(num_mask):
            x = utils.nhwc2nchw(x_new[num_batch*idx_iter:num_batch*(idx_iter+1)])
            y = utils.nhwc2nchw(y_new[num_batch*idx_iter:num_batch*(idx_iter+1)])
            x, y = torch.tensor(x), torch.tensor(y)
            x = x.to(self.config['device'])
            y = y.to(self.config['device'])

            if(training):
                self.optimizer.zero_grad()

                if(self.config['mode_lr'] == 0):
                    pass
                else:
                    if(epoch > 5 and epoch <= 250):
                        if(self.config['mode_lr'] == 1): # only warmup
                            tmp_lr = self.config['learning_rate']
                        elif(self.config['mode_lr'] == 2): # sgdr
                            if(self.lr_tmp == self.lr_min):
                                self.T_cur = 0
                                if(epoch == 0):
                                    pass
                                else:
                                    self.T_i *= self.T_mult

                            self.lr_tmp = \
                                self.lr_min + \
                                0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(self.T_cur / self.T_i * np.pi))
                            self.T_cur = \
                                int(self.T_cur) + \
                                (((iteration % iter_per_epoch)+1) / iter_per_epoch)
                            tmp_lr = self.lr_tmp
                        else: # non defined
                            tmp_lr = self.config['learning_rate']
                    elif(epoch > 250):
                        self.lr_tmp /= 10
                        tmp_lr = self.lr_tmp
                    else:
                        tmp_lr = (iteration * self.config['learning_rate']) / (iter_per_epoch*5)


                    # update learning rate
                    for pgroup in self.optimizer.param_groups:
                        pgroup['lr'] = tmp_lr

            step_dict = self.__model(x, y)
            y_hat = step_dict['y_hat']
            step_dict['y'] = y

            losses_tmp = {}
            losses_tmp['msgms_b'] = self.msgms(y, y_hat, as_loss=False)
            losses_tmp['msgms'] = torch.mean(losses_tmp['msgms_b'])

            tmp_ssim = lossf.loss_ssim(y, y_hat, (1, 2, 3))
            losses_tmp['ssim_b'] = torch.mean(tmp_ssim.reshape((m.shape[0], -1)), -1)
            losses_tmp['ssim'] = torch.mean(losses_tmp['ssim_b'])

            tmp_l2_b = lossf.loss_l2(y, y_hat, (1, 2, 3))
            losses_tmp['l2_b'] = torch.mean(tmp_l2_b.reshape((m.shape[0], -1)), -1)
            losses_tmp['l2'] = torch.mean(losses_tmp['l2_b'])

            losses_tmp['opt_b'] = torch.mean(losses_tmp['msgms_b'], (1, 2, 3)) + losses_tmp['ssim_b'] + losses_tmp['l2_b']
            losses_tmp['opt'] = torch.mean(losses_tmp['opt_b'])

            if(training):
                losses_tmp['opt'].backward()
                self.optimizer.step()

            losses['msgms_b'].append(utils.detach(losses_tmp['msgms_b']))
            losses['msgms'].append(utils.detach(losses_tmp['msgms']))
            losses['ssim'].append(utils.detach(losses_tmp['ssim']))
            losses['ssim_b'].append(utils.detach(losses_tmp['ssim_b']))
            losses['l2_b'].append(utils.detach(losses_tmp['l2_b']))
            losses['l2'].append(utils.detach(losses_tmp['l2']))
            losses['opt_b'].append(utils.detach(losses_tmp['opt_b']))
            losses['opt'].append(utils.detach(losses_tmp['opt']))
            y, y_hat = step_dict['y'], step_dict['y_hat']

            y = utils.detach(y)
            y_hat = utils.detach(y_hat)
            x = utils.detach(x)

            if(y_tot is None):
                y_tot, y_hat_tot, x_tot = y, y_hat, x
            else:
                y_tot = np.concatenate((y_tot, y), axis=0)
                y_hat_tot = np.concatenate((y_hat_tot, y_hat), axis=0)
                x_tot = np.concatenate((x_tot, x), axis=0)

        for idx_key, name_key in enumerate(losses.keys()):
            losses[name_key] = np.asarray(losses[name_key])
            if('_b' in name_key): continue
            losses[name_key] = np.average(losses[name_key])

        y, y_hat, x = y_tot, y_hat_tot, x_tot
        y = utils.nchw2nhwc(y)
        y_hat = utils.nchw2nhwc(y_hat)
        x = utils.nchw2nhwc(x)

        if(training):
            self.writer.add_scalar( \
                "%s/l2" %(self.__model.config['who_am_i']), \
                scalar_value=losses['l2'], global_step=iteration)
            self.writer.add_scalar( \
                "%s/ssim" %(self.__model.config['who_am_i']), \
                scalar_value=losses['ssim'], global_step=iteration)
            self.writer.add_scalar( \
                "%s/msgms" %(self.__model.config['who_am_i']), \
                scalar_value=losses['msgms'], global_step=iteration)
            self.writer.add_scalar( \
                "%s/opt" %(self.__model.config['who_am_i']), \
                scalar_value=losses['opt'], global_step=iteration)
            self.writer.add_scalar( \
                "%s/lr" %(self.__model.config['who_am_i']), \
                scalar_value=self.optimizer.param_groups[0]['lr'], global_step=iteration)

        # inverse masking
        m_b = np.expand_dims(m.reshape((m.shape[0]*m.shape[1], m.shape[2], m.shape[3])), axis=-1)
        y_hat_new = y_hat * m_b
        y_hat_new = y_hat_new.reshape((m.shape[0], m.shape[1], m.shape[2], m.shape[3], y_hat.shape[-1]))
        y_hat = np.average(y_hat_new, axis=1)
        losses['msgms_b'] = np.average(losses['msgms_b'].reshape((m.shape[0], m.shape[1], 1, m.shape[2], m.shape[3])), 1)
        map = utils.nchw2nhwc(losses['msgms_b'])

        return {'x_m':x, 'm':m, 'y':minibatch['x'], 'y_hat':y_hat, 'map':map, 'losses':losses}

    def save_params(self, model='base'):

        torch.save(self.__model.state_dict(), os.path.join(self.config['path_ckpt'], '%s.pth' %(model)))

    def load_params(self, model):

        self.__model.load_state_dict(torch.load(os.path.join(self.config['path_ckpt'], '%s' %(model))), strict=False)
        self.__model.eval()
