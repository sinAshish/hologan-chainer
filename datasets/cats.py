# code borrowed from https://github.com/AlexiaJM/RelativisticGAN/blob/master/code/preprocess_cat_dataset.py

import os
import glob
import math
import sys
import cv2
import numpy as np

from chainer import dataset


def rotateCoords(coords, center, angleRadians):
        # Positive y is down so reverse the angle, too.
        angleRadians = -angleRadians
        xs, ys = coords[::2], coords[1::2]
        newCoords = []
        n = min(len(xs), len(ys))
        i = 0
        centerX = center[0]
        centerY = center[1]
        cosAngle = math.cos(angleRadians)
        sinAngle = math.sin(angleRadians)
        while i < n:
                xOffset = xs[i] - centerX
                yOffset = ys[i] - centerY
                newX = xOffset * cosAngle - yOffset * sinAngle + centerX
                newY = xOffset * sinAngle + yOffset * cosAngle + centerY
                newCoords += [newX, newY]
                i += 1
        return newCoords

def preprocessCatFace(coords, image):

        leftEyeX, leftEyeY = coords[0], coords[1]
        rightEyeX, rightEyeY = coords[2], coords[3]
        mouthX = coords[4]
        if leftEyeX > rightEyeX and leftEyeY < rightEyeY and \
                        mouthX > rightEyeX:
                # The "right eye" is in the second quadrant of the face,
                # while the "left eye" is in the fourth quadrant (from the
                # viewer's perspective.) Swap the eyes' labels in order to
                # simplify the rotation logic.
                leftEyeX, rightEyeX = rightEyeX, leftEyeX
                leftEyeY, rightEyeY = rightEyeY, leftEyeY

        eyesCenter = (0.5 * (leftEyeX + rightEyeX),
                                  0.5 * (leftEyeY + rightEyeY))

        eyesDeltaX = rightEyeX - leftEyeX
        eyesDeltaY = rightEyeY - leftEyeY
        eyesAngleRadians = math.atan2(eyesDeltaY, eyesDeltaX)
        eyesAngleDegrees = eyesAngleRadians * 180.0 / math.pi

        # Straighten the image and fill in gray for blank borders.
        rotation = cv2.getRotationMatrix2D(
                        eyesCenter, eyesAngleDegrees, 1.0)
        imageSize = image.shape[1::-1]
        straight = cv2.warpAffine(image, rotation, imageSize,
                                                          borderValue=(128, 128, 128))

        # Straighten the coordinates of the features.
        newCoords = rotateCoords(
                        coords, eyesCenter, eyesAngleRadians)

        # Make the face as wide as the space between the ear bases.
        w = abs(newCoords[16] - newCoords[6])
        # Make the face square.
        h = w
        # Put the center point between the eyes at (0.5, 0.4) in
        # proportion to the entire face.
        minX = eyesCenter[0] - w/2
        if minX < 0:
                w += minX
                minX = 0
        minY = eyesCenter[1] - h*2/5
        if minY < 0:
                h += minY
                minY = 0

        # Crop the face.
        crop = straight[int(minY):int(minY+h), int(minX):int(minX+w)]
        #print(crop.shape)
        
        # Return the crop.
        return crop


class CatDataset(dataset.DatasetMixin):

    def __init__(self, dataroot, size=(64, 64)):
        super().__init__()

        self.dataroot = dataroot
        self.size = size
        
        self.elements = []
        
        for i in range(7):
            folder = "CAT_0{}".format(i)
            for imagePath in glob.glob(os.path.join(self.dataroot, folder, '*.jpg')):
                self.elements.append('%s.cat' % imagePath)
        #self.audit()
        print("loaded {} cat faces".format(len(self.elements)))

    def audit(self):
        # check each element
        _size = self.size
        self.size = None

        new_elements = []
        for i in range(len(self.elements)):
                #while i < total:
                img = self.get_example(i)
                print(i, len(self.elements), img.shape, img.size)
                if img.size > 0:
                        new_elements.append(self.get_example(i))
        self.elements = new_elements
        self.size = _size
        
    def __len__(self):
        return len(self.elements)

    def get_example(self, i):
        input = open(self.elements[i], 'r')
	# Read the coordinates of the cat features from the
	# file. Discard the first number, which is the number
	# of features.
        coords = [int(i) for i in input.readline().split()[1:]]
        imagePath = os.path.splitext(self.elements[i])[0]
        image = None
        #if self.size is not None:
        image = cv2.imread(imagePath, cv2.IMREAD_COLOR).astype(np.float32)
        img = preprocessCatFace(coords, image)
        if img is None:
                print >> sys.stderr, \
                        'Failed to preprocess image at %s.' % \
        	        imagePath
                exit()
        #print(img.shape, img.size)
        #print(self.elements[i])
        if self.size is not None:
                h, w = self.size
                #print(img.shape, img.size)
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        img /= 255.0
        img *= 2.0
        img -= 1.0

        return img.transpose(2, 0, 1)


if __name__ == "__main__":
    data = CatDataset("/mnt/netapp_vol01/calland/datasets/cats")
    img = data.get_example(10).transpose(1, 2, 0)
    img += 1
    img /= 2
    
    from matplotlib import pyplot as plt
    plt.imsave("img.png", img)
