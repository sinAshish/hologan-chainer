import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable


class GANUpdater(chainer.training.updaters.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self.loss = kwargs.pop("loss", "hinge")
        self.z_dim = kwargs.pop("z_dim", 128)
        self.style_disc = kwargs.pop("style_disc", True)
        self.id_loss = kwargs.pop("id_loss", True)
        self.template = None
        super(GANUpdater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, y_fake, y_real, s_fake, s_real, z_fake, z):
        batchsize = len(y_fake)

        if "lsgan" in self.loss:
            loss = F.sum(F.square(y_real - 1.)) + F.sum(F.square(y_fake))
            loss /= 2 * batchsize
        elif "hinge" in self.loss:
            loss = F.mean(F.relu(1. - y_real))
            loss += F.mean(F.relu(1. + y_fake))
        else:
            L1 = F.sum(F.softplus(-y_real)) / batchsize                                                                                                                                                                             
            L2 = F.sum(F.softplus(y_fake)) / batchsize
            loss = L1 + L2 

        if self.style_disc:
            #style_loss = F.mean(F.mean(F.relu(1.-s_real)) + F.mean(F.relu(1. + s_fake)))
            style_loss = F.sum(F.softplus(-s_real)) / batchsize + F.sum(F.softplus(s_fake)) / batchsize
        else:
            style_loss = 0.0

        if self.id_loss:
            identity_loss = F.mean_squared_error(z_fake, z)
        else:
            identity_loss = 0.
            
        chainer.report({'loss': loss, 'style_loss': style_loss, 'id_loss': identity_loss}, dis)
        return loss + style_loss + identity_loss

    def loss_gen(self, gen, y_fake, s_fake, z_fake, z):
        batchsize = len(y_fake)

        if "lsgan" in self.loss:
            loss = 0.5 * F.sum(F.square(y_fake - 1.)) / batchsize
        elif "hinge" in self.loss:
            loss = -F.mean(y_fake)
        else:
            loss = F.sum(F.softplus(-y_fake)) / batchsize
            
        if self.style_disc:
            #style_loss = F.mean(-F.mean(s_fake))
            style_loss = F.sum(F.softplus(-s_fake)) / batchsize
        else:
            style_loss = 0.0

        if self.id_loss:
            identity_loss = F.mean_squared_error(z_fake, z)
        else:
            identity_loss = 0.
            
        chainer.report({'loss': loss, 'style_loss': style_loss, 'id_loss': identity_loss}, gen)
        return loss + style_loss + identity_loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        batch = self.get_iterator('main').next()
        x_real  = self.converter(batch, self.device) #.get()
        batchsize = x_real.shape[0]

        xp = chainer.backend.get_array_module(x_real)

        gen, dis = self.gen, self.dis

        #z = chainer.Variable(xp.random.uniform(-1, 1, size=(batchsize, self.z_dim)).astype(np.float32))
        z = xp.random.uniform(-1, 1, size=(batchsize, self.z_dim)).astype(np.float32)

        #z = F.concat((z1, z2), axis=1)
        
        x_fake = gen(z, z)
        y_fake, z_fake, s_fake = dis(x_fake)
        gen_optimizer.update(self.loss_gen, gen, y_fake, s_fake, z_fake, z)
        y_real, z_real, s_real = dis(x_real)
        x_fake.unchain_backward()
        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real, s_real, s_fake, z_fake, z)
        #gen_optimizer.update(self.loss_gen, gen, y_fake, s_fake)
    
    def update_core2(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        batch = self.get_iterator('main').next()
        x_real  = self.converter(batch, self.device)
        batchsize = x_real.shape[0]
        
        xp = chainer.backend.get_array_module(x_real)

        # update gen
        gen, dis = self.gen, self.dis

        z1 = xp.random.uniform(-1, 1, size=(batchsize, self.z_dim)).astype(np.float32)
        #z2 = xp.random.uniform(-1, 1, size=(batchsize, self.z_dim)).astype(np.float32)
        
        x_fake = gen(z1, z1)
        y_fake, z_fake, s_fake = dis(x_fake)
        gen_optimizer.update(self.loss_gen, gen, y_fake, s_fake)

        for i in range(1):
            # update disc
            batch = self.get_iterator('main').next()
            x_real = self.converter(batch, self.device)
            z1 = xp.random.uniform(-1, 1, size=(batchsize, self.z_dim)).astype(np.float32)
            #z2 = xp.random.uniform(-1, 1, size=(batchsize, self.z_dim)).astype(np.float32)
            x_fake  = gen(z1, z1)

            y_real, z_real, s_real = dis(x_real)
            y_fake, z_fake, s_fake = dis(x_fake)
            #x_fake.unchain_backward()
            #s_fake.unchain() #_backward()
            dis_optimizer.update(self.loss_dis, dis, y_fake, y_real, s_real, s_fake)

