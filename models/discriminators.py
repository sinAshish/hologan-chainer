import numpy as np
import chainer

from chainer import functions as F
from chainer import links as L
from chainer import distributions as D

from common.sn_convolution_2d import SNConvolution2D
from common.sn_linear import SNLinear
from common.resblocks import Block, OptimizedBlock
from common.instance_normalization import InstanceNormalization


def _style_params(x):
    # check if this is a 2d or 3d embedding                                                                                                                                  
    idx = (-3, -2, -1) if len(x.shape) > 4 else (-2, -1)
    mu = F.mean(x, axis=idx, keepdims=True)
    std_dev = F.sqrt(F.mean((x - mu)**2, axis=idx))
    return F.concat([F.squeeze(mu), std_dev], axis=1)


class SNResNetProjectionDiscriminator(chainer.Chain):

    def __init__(self, ch=64, n_classes=0, activation=F.relu):
        super(SNResNetProjectionDiscriminator, self).__init__()
        self.activation = activation
        initializer = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.block1 = OptimizedBlock(None, ch)
            self.block2 = Block(ch, ch * 2, activation=activation, downsample=True)
            self.block3 = Block(ch * 2, ch * 4, activation=activation, downsample=True)
            self.block4 = Block(ch * 4, ch * 8, activation=activation, downsample=True)
            self.block5 = Block(ch * 8, ch * 16, activation=activation, downsample=True)
            self.l6 = SNLinear(ch * 16, 1, initialW=initializer)
            self.l_y = SNLinear(None, ch * 16, initialW=initializer)
            self.style_disc = StyleDiscriminator(5)
            
    def __call__(self, x, y=None):
        style = []
        h = x
        h = self.block1(h)
        style.append(_style_params(h))
        h = self.block2(h)
        style.append(_style_params(h))
        h = self.block3(h)
        style.append(_style_params(h))
        h = self.block4(h)
        style.append(_style_params(h))
        h = self.block5(h)
        style.append(_style_params(h))
        h = self.activation(h)
        h = F.sum(h, axis=(2, 3))  # Global pooling

        output = self.l6(h)
        if y is not None:
            w_y = self.l_y(y)
            output += F.sum(w_y * h, axis=1, keepdims=True)
        
        return output, self.style_disc(style)


class SNResNetConcatDiscriminator(chainer.Chain):
    def __init__(self, ch=64, n_classes=0, activation=F.relu, dim_emb=128):
        super(SNResNetConcatDiscriminator, self).__init__()
        self.activation = activation
        initializer = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.block1 = OptimizedBlock(3, ch)
            self.block2 = Block(ch, ch * 2, activation=activation, downsample=True)
            self.block3 = Block(ch * 2, ch * 4, activation=activation, downsample=True)
            self.l_y = SNLinear(None, dim_emb, initialW=initializer)
            self.block4 = Block(ch * 4 + dim_emb, ch * 8, activation=activation, downsample=True)
            self.block5 = Block(ch * 8, ch * 16, activation=activation, downsample=True)
            self.block6 = Block(ch * 16, ch * 16, activation=activation, downsample=False)
            self.l7 = SNLinear(ch * 16, 1, initialW=initializer)
            
    def __call__(self, x, y=None):
        #print(y.shape)
        #exit()
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        if y is not None:
            emb = self.l_y(y)
            H, W = h.shape[2], h.shape[3]
            emb = F.broadcast_to(
                F.reshape(emb, (emb.shape[0], emb.shape[1], 1, 1)),
                (emb.shape[0], emb.shape[1], H, W))
            h = F.concat([h, emb], axis=1)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.activation(h)
        h = F.sum(h, axis=(2, 3))  # Global pooling
        output = self.l7(h)
        return output


class StyleDiscriminator(chainer.Chain):

    def __init__(self, n_layers, n_hidden=1024):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        super(StyleDiscriminator, self).__init__()
        
        for i in range(self.n_layers):
            self.add_link("layer{}_0".format(i), L.Linear(None, self.n_hidden))
            self.add_link("layer{}_1".format(i), L.Linear(self.n_hidden, self.n_hidden))
            self.add_link("layer{}_2".format(i), L.Linear(self.n_hidden, 1))

    def __call__(self, x):
        assert len(x) == self.n_layers
        
        ret = []
        for i, f in enumerate(x):
            h = F.leaky_relu(self["layer{}_0".format(i)](f))
            h = F.leaky_relu(self["layer{}_1".format(i)](h))
            h = self["layer{}_2".format(i)](h)
            ret.append(h) #F.sigmoid(h))
        return F.concat(ret, axis=0)


class HoloGANDiscriminator(chainer.Chain):

    def __init__(self, identity_loss=True, z_dim=128):
        self.z_dim = z_dim
        self.identity_loss = identity_loss
        super().__init__()
        with self.init_scope():
            initial_w = chainer.initializers.Normal(scale=0.2)
            initial_b = chainer.initializers.Zero()
            self.conv_0 = SNConvolution2D(None, 128, ksize=3, stride=1, pad=1, initialW=initial_w, initial_bias=initial_b, nobias=True)
            self.conv_1 = SNConvolution2D(None, 256, ksize=3, stride=2, pad=1, initialW=initial_w, initial_bias=initial_b, nobias=True)
            self.conv_2 = SNConvolution2D(None, 512, ksize=3, stride=2, pad=1, initialW=initial_w, initial_bias=initial_b, nobias=True)
            self.conv_3 = SNConvolution2D(None, 1024, ksize=3, stride=2, pad=1, initialW=initial_w, initial_bias=initial_b, nobias=True)
            self.inst_0 = InstanceNormalization(128)
            self.inst_1 = InstanceNormalization(256)
            self.inst_2 = InstanceNormalization(512)
            self.inst_3 = InstanceNormalization(1024)
            
            self.out = SNLinear(None, 1, initialW=initial_w, initial_bias=initial_b)
            self.recon_0 = SNLinear(None, self.z_dim, initialW=initial_w, initial_bias=initial_b)
            self.recon_out = SNLinear(None, self.z_dim, initialW=initial_w, initial_bias=initial_b)

            self.style_disc = StyleDiscriminator(3)
            
    def __call__(self, x):
        style = []
        h = self.inst_0(self.conv_0(x))

        style.append(_style_params(h))
        h = self.inst_1(self.conv_1(F.leaky_relu(h)))

        style.append(_style_params(h))
        h = self.inst_2(self.conv_2(F.leaky_relu(h)))

        style.append(_style_params(h))
        h = self.inst_3(self.conv_3(F.leaky_relu(h)))

        h = F.leaky_relu(F.reshape(h, (h.shape[0], -1)))
        o = self.out(h)
        
        if self.identity_loss is False:
            return o, None, self.style_disc(style)
            
        # identity loss
        h = F.leaky_relu(self.recon_0(h))
        z = F.tanh(self.recon_out(h))
        return o, z, self.style_disc(style)
