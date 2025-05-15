import scipy.io as sio
import os
import numpy as np

"""
prepared for independent
"""

extrac_path = '/mnt/DATA-2/DEAP/Arousal/'
label_path = '/mnt/DATA-2/DEAP/Arousal/label.mat'
save_path = '/mnt/DATA-2/DEAP/2/'

dir_list = [f for f in os.listdir(extrac_path) if f != "label.mat"]

label = sio.loadmat(label_path)
label = label['label'][0]

for f in dir_list:
    """
    if '_' not in f:
        continue
    """
    S = sio.loadmat(extrac_path + f)
    DE = []
    labelAll = []
    for i in range(40):
        data = S['data' + str(i + 1)]
        if len(DE):
            DE = np.concatenate((DE, data), axis=1)
        else:
            DE = data

        if len(labelAll):
            labelAll = np.concatenate((labelAll, np.zeros([data.shape[1], 1]) + label[i]), axis = 0)
        else:
            labelAll = np.zeros([data.shape[1], 1]) + label[i]

    #print(DE.shape)
    #print(labelAll.shape)

    mdic = {"DE": DE, "labelAll": labelAll, "label": "experiment"}

    sio.savemat(save_path + f, mdic)
    print(extrac_path + f, '->', save_path + f)