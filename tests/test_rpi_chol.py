from sksparse.cholmod import cholesky
from scipy import sparse
import numpy as np
import time


if __name__ == "__main__":
    print("hello")
    N = 30000

    t1 = time.time()
    Q = sparse.eye(N).tocsc()
    
    t2 = time.time()
    print("Q", t2 - t1)

    t1 = time.time()
    Q_fac = cholesky(Q)
    t2 = time.time()
    print("Q_fac", t2 - t1)

    t1 = time.time()
    z = np.random.normal(size=N)
    y = Q_fac.apply_Pt(Q_fac.solve_Lt(z, use_LDLt_decomposition = False))
    t2 = time.time()
    print("z: " , t2 - t1)

