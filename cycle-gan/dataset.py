import six.moves.cPickle as Pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def loadImage(path, ARRAY=False):
    if not ARRAY:
        inImage_ = cv2.imread(path)
        inImage = cv2.cvtColor(inImage_, cv2.COLOR_RGB2BGR)
    else:
        inImage = path
    info = np.iinfo(inImage.dtype)
    inImage = inImage.astype(np.float) / info.max

    iw = inImage.shape[1]
    ih = inImage.shape[0]
    if iw < ih:
        inImage = cv2.resize(inImage, (64, int(64 * ih / iw)))
    else:
        inImage = cv2.resize(inImage, (int(64 * iw / ih), 64))
    inImage = inImage[0:64, 0:64]
    inImage = inImage[0:64, 0:64]
    inImage = cv2.resize(inImage, (64, 64))
    return inImage


def plot(samples):
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(3, 10)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        sample = (sample) * 255
        sample = sample.astype(np.uint8)
        plt.imshow(sample)

    return fig


class MangoDataset():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        with open('table.pkl', 'rb') as cloth:
            self.cloth_table = Pickle.load(cloth)

        self.cn = len(self.cloth_table)
        self.path = data_dir
        self.size = 64
        self.channel = 3

    def getbatch(self, batchsize=64):
        img = []

        for i in range(batchsize):
            r1 = int(np.random.randint(0, self.cn, (1,)).item())
            path3 = self.cloth_table[r1]
            img3 = loadImage(self.path + path3)
            img.append(img3)
        return np.array(img)