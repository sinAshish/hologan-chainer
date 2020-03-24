import os
import numpy as np
import cv2

from chainer import dataset


class CarsDataset(dataset.DatasetMixin):

    def __init__(self, data_root, size=(64, 64)):
        self.data_root = data_root
        self.size = size
        self.data = []
        for filename in os.listdir(self.data_root):
            if filename.endswith(".jpg"):
                self.data.append(filename)

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        filepath = os.path.join(self.data_root, self.data[i])
        img = cv2.imread(filepath, cv2.IMREAD_COLOR).astype(np.float32)
        h, w = self.size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        img /= 255.0
        img *= 2.0
        img -= 1.0

        return img.transpose(2, 0, 1)

        
if __name__ == "__main__":
    datapath = "/mnt/netapp_vol01/calland/datasets/cars/cars_train"
    dataset = CarsDataset(datapath)
    img = dataset.get_example(10)
    print(img.shape, img.min(), img.max())

    import matplotlib.pyplot as plt
    img += 1.
    img /= 2.
    plt.imsave("car.png", img.transpose(1, 2, 0))
