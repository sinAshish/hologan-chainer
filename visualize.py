#!/usr/bin/env python

import os
import numpy as np
from PIL import Image
import cv2

import chainer
import chainer.backends.cuda
from chainer import Variable
from chainer import functions as F

from models.generators import HoloGANGenerator, HoloGANGenerator2


def make_image(gen, rows, cols, z1, z2, theta=None):
    #print(theta)
    #gen.xp.random.seed(seed)
    n_images = rows * cols
    xp = gen.xp
    
    with chainer.using_config('enable_backprop', False), chainer.using_config('train', False):
        #print(theta)
        x = gen(z1, z2, theta=theta)
            
    x = chainer.backends.cuda.to_cpu(x.array)
    #np.random.seed()
    
    x += 1.0
    x /= 2.0
    x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
    B, C, H, W = x.shape
    
    x = x.reshape((rows, cols, C, H, W))
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((rows * H, cols * W, C))

    # make progress bar for theta
    """if theta is not None:
        max_len = rows*W
        current = int(((theta-220) / 100.) * max_len)
        x[:3, current] = 0
    """
    return x
        

def out_generated_image(gen, dis, rows, cols, seed, dst):
        
    @chainer.training.make_extension()
    def save_image(trainer, theta=None):
        z1 = gen.xp.random.uniform(-1, 1, size=(rows*cols, 128)).astype(np.float32)
        z2 = gen.xp.random.uniform(-1, 1, size=(rows*cols, 128)).astype(np.float32)

        x = make_image(gen, rows, cols, z1, z1, theta=theta)

        # save to disk
        preview_dir = '{}/preview'.format(dst)
        if trainer is not None:
            preview_path = preview_dir +\
                           '/image{:0>8}.png'.format(trainer.updater.iteration)
        else:
            preview_path = preview_dir + "preview.png"
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        cv2.imwrite(preview_path, x)
        
    return save_image


def make_animation(gen):
    import imageio

    rows = 5
    cols = 5
    gen.xp.random.seed(100)
    z1 = gen.xp.random.uniform(-1, 1, size=(rows*cols, 128)).astype(np.float32)
    #print(z1)    #z1 = gen.xp.broadcast_to(z1, (rows * cols, 128))
    z2 = gen.xp.random.uniform(-1, 1, size=(rows*cols, 128)).astype(np.float32)
    #z2 = gen.xp.broadcast_to(z2, (rows * cols, 128))

    images = []
    for th in np.arange(-50, 50, 1): #(-180., 180., 5.):
        theta = gen.xp.array([th] * z1.shape[0], dtype=np.float32)
        #theta = None
        #print(theta)
        img = make_image(gen, rows, cols, z1, z1, theta=theta)[:,:,::-1]
        images.append(img)

    imageio.mimsave("./cats.gif", images, fps=10)
        
    #imageio.mimsave('./cats.gif', [make_image(gen, rows, cols, z1, z1, np.broadcast_to(th, z1.shape[0]))[:,:,::-1] for th in np.arange(220., 320., 1.)], fps=10)
    
    
    
if __name__ == "__main__":
    device = 0
    #modelpath = "results_3/gen_epoch_650.npz"
    #modelpath = "results_5/gen_epoch_50.npz"
    #modelpath = "results_postwar_cats/gen_epoch_1.npz"
    #modelpath = "results_postwar_cats_rotation/gen_epoch_1000.npz"
    #modelpath = "results_cats_sagan_rotations_styledisc_idloss_b0.09/gen_epoch_350.npz"
    #modelpath = "results_cats_sagan_rotations_reweight/gen_epoch_420.npz"
    #modelpath = "results_cats_sagan_rotations_nn/gen_epoch_720.npz"
    modelpath = "output/trilinear_zrot/gen_epoch_840.npz"
    gen = HoloGANGenerator2(theta_range=(-50., 50))#(220., 320))
    chainer.serializers.load_npz(modelpath, gen)

    
    chainer.backends.cuda.get_device_from_id(device).use()
    gen.to_gpu()

    make_animation(gen)
    
    #func = out_generated_image(gen, None, 5, 5, 0, "output")
    #func(None)
