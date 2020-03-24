import argparse
import numpy as np
import os
import cv2
import chainer
from chainer import training
from chainer.training import extensions

from datasets.cats import CatDataset
from datasets.cars import CarsDataset
from datasets.celeba import CelebADataset
from models.generators import HoloGANGenerator2
from models.discriminators import SNResNetProjectionDiscriminator, SNResNetConcatDiscriminator, HoloGANDiscriminator
from updater import GANUpdater

from visualize import out_generated_image

def main(args):
    epoch = args.epochs
    adam_decay_epoch = args.decay
    out = args.save_dir
    device = args.gpu
    batchsize = args.batch_size
    style_disc = args.use_style
    
    if args.dataset == 'cats':
        dataset= CatDataset(os.path.join(args.image_dir, 'cats'))
    elif args.dataset == 'cars':
        dataset = CarsDataset(os.path.join(args.image_dir, 'cars')) 
    else:
        dataset = CelebADataset(os.path.join(args.image_dir, 'celeba'))
        
    _range = (-50., 50.) #(220., 320) #cat_range
    dis = HoloGANDiscriminator() #SNResNetProjectionDiscriminator()
    gen = HoloGANGenerator2(theta_range=_range)
    
    chainer.backends.cuda.get_device_from_id(device).use()
    gen.to_gpu()
    dis.to_gpu()

     # Setup an optimizer
    def make_optimizer(model, alpha=1.E-4, beta1=0., beta2=0.999):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        #optimizer = chainer.optimizers.RMSprop(lr=1.0E-4)
        optimizer.setup(model)
        #optimizer.add_hook(
        #    chainer.optimizer_hooks.WeightDecay(1.E-4), 'hook_dec')
        return optimizer

    opt_gen = make_optimizer(gen, alpha= args.gen_lr, beta1=0., beta2=0.9)
    opt_dis = make_optimizer(dis, alpha= args.disc_lr, beta1=0., beta2=0.9)

    #train_iter = chainer.iterators.SerialIterator(dataset, batchsize)
    train_iter = chainer.iterators.MultiprocessIterator(dataset, batchsize, n_processes=8)

    #setup the updater
    updater = GANUpdater(
        models=(gen, dis),
        iterator=train_iter,
        optimizer={
            'gen': opt_gen, 'dis': opt_dis},
        device=device,
        style_disc=style_disc)

    # Setup a trainer
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

    snapshot_interval = (args.snapshot_interval, 'epoch')
    visualize_interval = (args.visualize_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    
    trainer.extend(
        extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_epoch_{.updater.epoch}.npz'), trigger= snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_epoch_{.updater.epoch}.npz'), trigger= snapshot_interval)
    trainer.extend(extensions.LogReport(trigger= display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/loss', 'dis/loss', 'gen/style_loss',
         'gen/id_loss', 'dis/style_loss']), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval= args.update_interval))
    trainer.extend(
        out_generated_image(
            gen, dis,
            5, 5, 0, out),
        trigger= visualize_interval)

    if args.resume is not None:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.extend(extensions.ExponentialShift("alpha", 0.5, optimizer=opt_gen),
                   trigger=(adam_decay_epoch, 'epoch'))
    trainer.extend(extensions.ExponentialShift("alpha", 0.5, optimizer=opt_dis),
                   trigger=(adam_decay_epoch, 'epoch'))

    # Run the training
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default = 2000, type = int)
    parser.add_argument('--decay', default = 1000, type = int)
    parser.add_argument('--save_dir', default='output/rot', type = str)
    parser.add_argument('--gpu', default= 0, type = int)
    parser.add_argument('--batch_size', default= 64, type= int)
    parser.add_argument('--use_style', default= False, action= 'store_true')
    parser.add_argument('--gen_lr', default=1e-4, type = float)
    parser.add_argument('--disc_lr', default=4e-4, type = float)
    parser.add_argument('--image_dir', default='data/', type = str)
    parser.add_argument('--snapshot_interval', default=10, type = int)
    parser.add_argument('--display_interval', default= 10, type = int)
    parser.add_argument('--visualize_interval', default= 100, type = int)
    parser.add_argument('--update_interval', default= 10, type= int)
    parser.add_argument('--dataset', default= 'cats', type= str)
    parser.add_argument('--resume', type=str, default='')
    args = parser.parse_args()
    main(args)
