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
        self.config = {}
        self.config['nn'] = utils.key_parsing(kwargs, 'nn', 1000)

        self.config['dim_h'] = utils.key_parsing(kwargs, 'dim_h', 28)
        self.config['dim_w'] = utils.key_parsing(kwargs, 'dim_w', 28)
        self.config['dim_c'] = utils.key_parsing(kwargs, 'dim_c', 1)
        self.config['ksize'] = utils.key_parsing(kwargs, 'ksize', 3)

        self.config['mode_optim'] = utils.key_parsing(kwargs, 'mode_optim', 'sgd')
        self.config['learning_rate'] = utils.key_parsing(kwargs, 'learning_rate', 1e-3)
        self.config['mode_lr'] = utils.key_parsing(kwargs, 'mode_lr', 0)

        self.config['path_ckpt'] = utils.key_parsing(kwargs, 'path_ckpt', 'Checkpoint')
        self.config['ngpu'] = utils.key_parsing(kwargs, 'ngpu', 1)
        self.config['device'] = utils.key_parsing(kwargs, 'device', 'cuda')

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
        x_new, y_new = [], []
        for idx_x in range(x.shape[0]):
            images, maskes = utils.masking(x[idx_x].copy(), m[idx_x].copy(), inverse=False)
            x_new.extend(maskes)
            y_new.extend(images)
        x_new = np.asarray(x_new).astype(np.float32)
        y_new = np.asarray(y_new).astype(np.float32)
        x, y = x_new, y_new

        x, y = utils.nhwc2nchw(x), utils.nhwc2nchw(y)
        x, y = torch.tensor(x), torch.tensor(y)
        x = x.to(self.config['device'])
        y = y.to(self.config['device'])

        if(training):
            self.optimizer.zero_grad()

            if(self.config['mode_lr'] == 0):
                pass
            else:
                if(epoch > 5):
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
                else:
                    tmp_lr = (iteration * self.config['learning_rate']) / (iter_per_epoch*5)

                # update learning rate
                for pgroup in self.optimizer.param_groups:
                    pgroup['lr'] = tmp_lr

        step_dict = self.__model(x, y)
        y_hat = step_dict['y_hat']
        step_dict['y'] = y


        losses = {}
        tmp_msgms = self.msgms(y, y_hat, as_loss=False)
        losses['msgms_b'] = torch.mean(tmp_msgms.reshape((m.shape[0], m.shape[1], self.config['dim_c'], m.shape[2], m.shape[3])), 1)
        losses['msgms'] = torch.mean(losses['msgms_b'])

        tmp_ssim = lossf.loss_ssim(y, y_hat, (1, 2, 3))
        losses['ssim_b'] = torch.mean(tmp_ssim.reshape((m.shape[0], -1)), -1)
        losses['ssim'] = torch.mean(losses['ssim_b'])

        tmp_l2_b = lossf.loss_l2(y, y_hat, (1, 2, 3))
        losses['l2_b'] = torch.mean(tmp_l2_b.reshape((m.shape[0], -1)), -1)
        losses['l2'] = torch.mean(losses['l2_b'])

        losses['opt_b'] = torch.mean(losses['msgms_b'], (1, 2, 3)) + losses['ssim_b'] + losses['l2_b']
        losses['opt'] = torch.mean(losses['opt_b'])

        if(training):
            losses['opt'].backward()
            self.optimizer.step()

        y, y_hat = step_dict['y'], step_dict['y_hat']

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

        for key in list(losses.keys()):
            try: losses[key] = losses[key].detach().numpy()
            except: losses[key] = losses[key].cpu().detach().numpy()

        try: y_hat = utils.nchw2nhwc(y_hat.detach().numpy())
        except: y_hat = utils.nchw2nhwc(y_hat.cpu().detach().numpy())

        # inverse masking
        m_b = np.expand_dims(m.reshape((m.shape[0]*m.shape[1], m.shape[2], m.shape[3])), axis=-1)
        y_hat_new = y_hat * m_b
        y_hat_new = y_hat_new.reshape((m.shape[0], m.shape[1], m.shape[2], m.shape[3], y_hat.shape[-1]))
        y_hat = np.average(y_hat_new, axis=1)

        map = utils.nchw2nhwc(losses['msgms_b'])

        return {'y':minibatch['x'], 'y_hat':y_hat, 'map':map, 'losses':losses}

    def save_params(self, model='base'):

        torch.save(self.__model.state_dict(), os.path.join(self.config['path_ckpt'], '%s.pth' %(model)))

    def load_params(self, model):

        self.__model.load_state_dict(torch.load(os.path.join(self.config['path_ckpt'], '%s' %(model))), strict=False)
        self.__model.eval()
