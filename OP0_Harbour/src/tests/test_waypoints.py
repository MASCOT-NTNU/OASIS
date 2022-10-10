import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
waypoints = np.array([[41.18505, -8.70509, 0],
[41.18476, -8.70541, 0.5],
[41.18448, -8.70508, 0],
[41.18475, -8.70475, 0]
])

print(waypoints)
N = 40
wp = np.empty([0, 3])
for i in range(N):
    for j in range(len(waypoints)):
        wp = np.append(wp, waypoints[j].reshape(1, -1), axis=0)
df = pd.DataFrame(wp, columns=['lat', 'lon', 'depth'])
df.to_csv("waypoints.csv", index=False)

print("wp: ", wp)
plt.plot(wp[:, 1], wp[:, 0], 'k.-')
plt.show()
