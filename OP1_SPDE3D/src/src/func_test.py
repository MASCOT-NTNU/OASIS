from WGS import WGS
import pandas as pd

import numpy as np

box = pd.read_csv('box.csv').to_numpy()

x, y = WGS.latlon2xy(box[:, 0], box[:, 1])
box_xy = np.vstack((x, y )).T
df = pd.DataFrame(box_xy, columns=['x', 'y'])
df.to_csv("box_xy.csv", index=False)

