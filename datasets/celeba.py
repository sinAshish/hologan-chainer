import os
import numpy as np
import cv2
from chainer import dataset
import chainer

class CelebADataset(dataset.DatasetMixin):

    def __init__(self, paths, root='.'):
        self.base = chainer.datasets.ImageDataset(paths, root)

        self.images=paths
        self.root=root

        self.root=root
        self.files=paths

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image = self.base[i]
        #import pdb; pdb.set_trace()
        return image
