import matplotlib.pyplot as plt
import numpy as np



def cut2D3(seismic, patch_size, stride):
    #0-1
    seismic = seismic - np.min(seismic)
    seismic = seismic / abs(seismic).max()
    im = seismic
    plt.imshow(im, cmap="gray",aspect='auto')
    plt.colorbar()
    plt.show()
    print(im.shape)
    data = np.pad(im, patch_size // 2, 'constant', constant_values=0)
    label_allset = []
    for i in range(0, seismic.shape[0], stride):
        label_set = []
        for j in range(0, seismic.shape[1], stride):
            label = data[i:i + patch_size, j:j + patch_size]
            label_set.append(label)
        label_allset.append(np.array(label_set))
    label_allset = np.array(label_allset).reshape((-1, patch_size, patch_size))

    print(label_allset.shape)
    return label_allset


if __name__ == '__main__':

    patch_size = 64
    stride = patch_size//32 #p
    #load seismic section
    seismic = np.load(r"z3-z4.npy")
    seismicdataset = cut2D3(seismic, patch_size, stride)
    print(seismicdataset.shape)
    np.save(r'z3-z4dataset.npy', seismicdataset)
