import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipdb

def filter_high_loss(num_filter=100):
    dat = pd.read_csv('loss_nuro.txt', header=None, sep=' ')
    dat = dat.replace([np.inf, -np.inf], np.nan)
    dat_dropna = dat.dropna(subset=[1])
    dat_dropna = dat_dropna.sort_values(by=[1])
    high_loss_images = dat_dropna[0].to_list()[-num_filter:]
    # high_loss_indices = (dat_dropna.index).to_numpy()
    return high_loss_images

if __name__ == '__main__':
    hl_idx = filter_high_loss() 
