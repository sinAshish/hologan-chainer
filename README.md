# HoloGAN

This is an attempt to implement HoloGAN in chainer.

## Abstract

They propose a novel GAN for the task of unsupervised learning of 3D representations from natural images. Most generative models rely on 2D kernels to generate images and make few assumptions about the 3D world. These models therefore tend to create blurry images or artefacts in tasks that require a strong 3D understanding, such as novel-view synthesis. HoloGAN instead learns a 3D representation of the world, and to render this representation in a realistic manner. Unlike other GANs, HoloGAN provides explicit control over the pose of generated objects through rigid-body transformations of the learnt 3D features. Our experiments show that using explicit 3D features enables HoloGAN to disentangle 3D pose and identity, which is further decomposed into shape and appearance, while still being able to generate images with similar or higher visual quality than other generative models. HoloGAN can be trained end-to-end from unlabelled 2D images only. Particularly, we do not require pose labels, 3D shapes, or multiple views of the same objects. This shows that HoloGAN is the first generative model that learns 3D representations from natural images in an entirely unsupervised manner.

![model](images/model.png)
## Dataset

I've used Cats dataset for training purposes here due to it's size.
The dataloaders for Cats, Cars and CelebA is provided here as well.

## Preprocessing

Put the dataset in the data folder.
The folder structure is shown below:

- data 
  - cats/
  - cars/
  - celeba/

## Training

To run the training code, simply run:

```highlight bash
python train.py --epochs 2000 --save_dir 'output' --batch_size 64 --use_style --dataset 'cats' 
```

## Results

![cats](images/cats.png)


NOTE : This code was written during my internship period as a trial project before the authors had published their code, so there would be some inconsistencies with the [Author's implementation](https://github.com/thunguyenphuoc/HoloGAN).

## Citation

If you find this useful in your work. Consider citing the authors work
