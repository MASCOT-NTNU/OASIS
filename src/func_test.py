import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

op = np.loadtxt("OperationArea.txt", delimiter=', ')

plt.plot(op[:, 1], op[:, 0], 'r-.')
plt.show()

df = pd.DataFrame(op, columns=['lat', 'lon'])
df.to_csv("OperationArea.csv", index=False)

from src.WGS import WGS

x, y = WGS.latlon2xy(op[:, 0], op[:, 1])
oop = np.stack((x, y), axis=1)

#%%
from shapely.geometry import Polygon
p = Polygon(oop)
print("Area: ", p.area/1e6)

plt.plot(oop[:, 1], oop[:, 0], 'k.-')
plt.show()


