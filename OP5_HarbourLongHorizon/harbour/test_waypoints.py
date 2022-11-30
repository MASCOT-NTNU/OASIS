import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
waypoints = pd.read_csv("polygon_test_wrongdate.csv").to_numpy()
plg =np.loadtxt("polygon.txt")
# waypoints = np.array([
# 	[41.18570285, -8.7057614, 0.0],
# 	[41.18552749, -8.70614696,  0.0],
# 	[41.18486741, -8.70571513, 0.0],
# 	[41.18469179,  -8.70601822, 0.0],
# 	[41.18431027, -8.7056132, 0.5],
# 	[41.18460297, -8.7053128, 0.5],
# 	[41.18452627, -8.70450545,  0.0],
# 	[41.18484521, -8.70524842, 0.5],
# 	[41.185366, - 8.70542277, 0.5]
# ])
print(plg)

# print(waypoints)
N = 40
wp = np.empty([0, 3])
for i in range(N):
    for j in range(len(waypoints)):
        wp = np.append(wp, waypoints[j].reshape(1, -1), axis=0)
df = pd.DataFrame(wp, columns=['lat', 'lon', 'depth'])
df.to_csv("waypoints.csv", index=False)

# print("wp: ", wp)
plt.plot(wp[:, 1], wp[:, 0], 'k.-')
plt.plot(plg[:, 1], plg[:, 0], 'r-.')
plt.show()
