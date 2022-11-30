import numpy as np
from scipy.spatial.distance import cdist
import time
from scipy.stats import norm

n1 = 100
n2 = 70

xv = np.linspace(0, 1, n1)
yv = np.linspace(0, 1, n2)
xx, yy = np.meshgrid(xv, yv)
x = xx.reshape(-1, 1)
y = yy.reshape(-1, 1)
N = len(x)
print("N: ", N)

# set matern kernel coefficients
sigma = 1
phi = .001
nugget = .001
R = np.diagflat(nugget)

mu_cond = x**2 + y**2
grid = np.hstack((x, y))
dm = cdist(grid, grid)
Sigma_cond = sigma**2 * (1 + phi * dm) * np.exp(-phi * dm)

salinity_measured = np.diagflat(10)

t1 = time.time()
F = np.zeros([N, 1])
F[0] = True
C = F.T @ Sigma_cond @ F + R
mu_cond = mu_cond + Sigma_cond @ F @ np.linalg.solve(C, (salinity_measured - F.T @ mu_cond))
Sigma_cond = Sigma_cond - Sigma_cond @ F @ np.linalg.solve(C, F.T @ Sigma_cond)
t2 = time.time()
print("update takes: ", t2 - t1)


t1 = time.time()
p = norm.cdf(.5, mu_cond, Sigma_cond)
bv = p * (1 - p)
ibv = np.sum(bv)
t2 = time.time()
print("EIBV takes: ", t2 - t1)
