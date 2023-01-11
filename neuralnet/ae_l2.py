import torch
import torch.nn as nn
import torch.nn.functional as F

import source.utils as utils

class Neuralnet(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.config = {}
        self.config['who_am_i'] = "U-Net_NS"

        self.config['dim_h'] = utils.key_parsing(kwargs, 'dim_h', 28)
        self.config['dim_w'] = utils.key_parsing(kwargs, 'dim_w', 28)
        self.config['dim_c'] = utils.key_parsing(kwargs, 'dim_c', 1)
        self.config['ksize'] = utils.key_parsing(kwargs, 'ksize', 3)

        self.config['filters'] = utils.key_parsing(kwargs, 'filters', None)
        self.config['device'] = utils.key_parsing(kwargs, 'device', 'cuda')

        self.filters_enc = self.config['filters']
        self.filters_dec = self.config['filters'][::-1][:-1]

        self.encoder, self.skip_enc = [], []
        for idx_enc, _ in enumerate(self.filters_enc[:-1]):
            if(idx_enc == 0): continue
            self.encoder.append( \
                ConvLayer(self.filters_enc[idx_enc-1], self.filters_enc[idx_enc], self.config['ksize'], \
                stride=1, name='enc_%d_1' %(idx_enc)).to(self.config['device']))
            self.skip_enc.append(False)
            self.encoder.append( \
                ConvLayer(self.filters_enc[idx_enc], self.filters_enc[idx_enc], self.config['ksize'], \
                stride=1, name='enc_%d_2' %(idx_enc)).to(self.config['device']))
            self.skip_enc.append(True)
            self.encoder.append( \
                ConvLayer(self.filters_enc[idx_enc], self.filters_enc[idx_enc], self.config['ksize'], \
                stride=2, name='enc_%d_3' %(idx_enc)).to(self.config['device']))
            self.skip_enc.append(False)
        self.encoder.append( \
            ConvLayer(self.filters_enc[-2], self.filters_enc[-1], self.config['ksize'], \
            stride=1, name='enc_%d_1' %(len(self.filters_enc))).to(self.config['device']))
        self.skip_enc.append(False)
        self.encoder.append( \
            ConvLayer(self.filters_enc[-1], self.filters_enc[-1], self.config['ksize'], \
            stride=1, name='enc_%d_2' %(len(self.filters_enc))).to(self.config['device']))
        self.skip_enc.append(False)

        self.decoder, self.skip_dec = [], []
        for idx_dec, _ in enumerate(self.filters_dec):
            if(idx_dec == 0): continue
            self.decoder.append( \
                UpConvLayer(self.filters_dec[idx_dec-1], self.filters_dec[idx_dec], self.config['ksize'], \
                stride=1, name='dec_%d_1' %(idx_dec)).to(self.config['device']))
            self.skip_dec.append(True)
            self.decoder.append( \
                ConvLayer(self.filters_dec[idx_dec]*2, self.filters_dec[idx_dec], self.config['ksize'], \
                stride=1, name='dec_%d_2' %(idx_dec)).to(self.config['device']))
            self.skip_dec.append(False)
            self.decoder.append( \
                ConvLayer(self.filters_dec[idx_dec], self.filters_dec[idx_dec], self.config['ksize'], \
                stride=1, name='dec_%d_3' %(idx_dec)).to(self.config['device']))
            self.skip_dec.append(False)
        self.decoder.append( \
            ConvLayer(self.filters_dec[-1], self.config['dim_c'], self.config['ksize'], \
            stride=1, batch_norm=False, activation=None, name='dec_%d_out' %(len(self.filters_dec))).to(self.config['device']))
        self.skip_dec.append(False)

        self.total = []
        for idx_enc, _ in enumerate(self.encoder):
            self.total.append(self.encoder[idx_enc])
        for idx_dec, _ in enumerate(self.decoder):
            self.total.append(self.decoder[idx_dec])
        self.modules = nn.ModuleList(self.total)

    def forward(self, x, y):

        skips = []
        for idx_enc, _ in enumerate(self.encoder):
            x = self.encoder[idx_enc](x)
            if(self.skip_enc[idx_enc]):
                skips.append(x)
        skips = skips[::-1]
        z = x

        idx_skip = 0
        for idx_dec, _ in enumerate(self.decoder):
            tmp_x = self.decoder[idx_dec](x)
            if(self.skip_dec[idx_dec]):
                cut_h, cut_w = skips[idx_skip].shape[-2], skips[idx_skip].shape[-1]
                x = tmp_x[:, :, :cut_h, :cut_w]
                idx_skip += 1
            else:
                x = tmp_x
        y_hat = torch.clamp(x, min=1e-12, max=1-(1e-12))

        return {'y_hat':y_hat, 'z':z}

class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=2, batch_norm=True, activation='lrelu', name=""):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module("%s_conv" %(name), nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2))
        if(batch_norm): self.conv.add_module("%s_bn" %(name), nn.BatchNorm2d(out_channels))
        if(activation=='sigmoid'): self.conv.add_module("%s_act" %(name), nn.Sigmoid())
        elif(activation=='lrelu'): self.conv.add_module("%s_act" %(name), nn.LeakyReLU())

    def forward(self, x):

        return self.conv(x)

class UpConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=True, activation='lrelu', name=""):
        super().__init__()

        self.upconv = nn.Sequential()
        self.upconv.add_module("%s_upsample" %(name), nn.UpsamplingNearest2d(scale_factor=2))
        self.upconv.add_module("%s_conv" %(name), nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2))
        if(batch_norm): self.upconv.add_module("%s_bn" %(name), nn.BatchNorm2d(out_channels))
        if(activation=='sigmoid'): self.upconv.add_module("%s_act" %(name), nn.Sigmoid())
        elif(activation=='lrelu'): self.upconv.add_module("%s_act" %(name), nn.LeakyReLU())

    def forward(self, x):

        n, c, h, w = x.shape
        return self.upconv(x)[:, :, :h*2, :w*2]
