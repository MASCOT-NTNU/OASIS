import h5py
import time
import numpy as np
import pandas as pd

plg_path = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Projects/OASIS/OperationArea.csv"

plg = pd.read_csv(plg_path).to_numpy()

# plt.plot(plg[:, 1], plg[:, 0], 'k.-')
# plt.show()

datapath = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Delft3D/raw/Nov2016_sal_1.mat"


class Delft3D:
    """

    """


