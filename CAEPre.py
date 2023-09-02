import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from CAEL1 import CAE


def patch_extractor2D(img, mid_x, mid_y, patch_size, dimensions=1):
    try:
        x, y, c = img.shape
    except ValueError:
        x, y = img.shape
        c = 1
    patch = np.pad(img, patch_size // 2, 'constant', constant_values=0)[int(mid_y):int(mid_y + patch_size),
            int(mid_x):int(mid_x + patch_size)]
    if c != dimensions:
        tmp_patch = np.zeros((patch_size, patch_size, dimensions))
        for uia in range(dimensions):
            tmp_patch[uia, :] = patch
        return tmp_patch
    return patch



#load seismic section
pre_data = np.load(r'z3-z4.npy')


def predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t_max, x_max = pre_data.shape
    #Load network
    model = torch.load(r'z3-z4(L1mse0.002).pth')
    model = model.to(device)
    patch_size = 64
    pre = None


    ######提取整个剖面的隐含变量
    for space in range(x_max):
        for depth in range(t_max):
            img = np.expand_dims(patch_extractor2D(pre_data, space, depth, patch_size, 1), axis=0)

            img = torch.from_numpy(img)
            img = img.type(torch.FloatTensor)

            img = torch.unsqueeze(img, dim=0)
            img_ = img.cuda()

            with torch.no_grad():
                z, x_reconst = model(img_)

                z = torch.squeeze(z, dim=1)
                z = z.reshape(z.shape[0], -1)

                output = z.detach().cpu().numpy()

                if pre is None:
                    pre = output
                else:
                    pre = np.vstack([pre, output])
                print(pre.shape)

    #save the extracted latent eigenvalues
    np.save(r'z3-z4pre.npy', pre)
    return pre



if __name__ == '__main__':

    pre = predict()