import mat73
import time
import numpy as np


class Delft3D:
    """

    """

    def reorganise_sinmod_data(self):
        t1 = time.time()
        self.data_sinmod = []
        for i in range(self.lat_sinmod.shape[0]):
            for j in range(self.lat_sinmod.shape[1]):
                for k in range(len(self.depth_sinmod)):
                    self.data_sinmod.append([self.lat_sinmod[i, j], self.lon_sinmod[i, j],
                                        self.depth_sinmod[k], self.salinity_sinmod_average[k, i, j]])
        self.data_sinmod = np.array(self.data_sinmod)
        t2 = time.time()
        print("Finished data reorganising... Time consumed: ", t2 - t1)

    def get_data_at_coordinates(self, coordinates, filename=False):
        # self.pool = mp.Pool(3)
        print("Start interpolating...")
        self.reorganise_sinmod_data()
        lat_sinmod = self.data_sinmod[:, 0]
        lon_sinmod = self.data_sinmod[:, 1]
        depth_sinmod = self.data_sinmod[:, 2]
        salinity_sinmod = self.data_sinmod[:, 3]

        print("Coordinates shape: ", coordinates.shape)
        self.lat_coordinates = coordinates[:, 0]
        self.lon_coordinates = coordinates[:, 1]
        self.depth_coordinates = coordinates[:, 2]
        ts = time.time()
        x_coordinates, y_coordinates = latlon2xy(self.lat_coordinates, self.lon_coordinates, 0, 0)
        x_sinmod, y_sinmod = latlon2xy(lat_sinmod, lon_sinmod, 0, 0)
        x_coordinates, y_coordinates, depth_coordinates, x_sinmod, y_sinmod, depth_sinmod = \
            map(vectorise, [x_coordinates, y_coordinates, self.depth_coordinates, x_sinmod, y_sinmod, depth_sinmod])
        print("Launching multiprocessing")
        t1 = time.time()
        # dm_x = self.pool.apply_async(get_distance_matrix, args=(x_coordinates, x_sinmod))
        # dm_y = self.pool.apply_async(get_distance_matrix, args=(y_coordinates, y_sinmod))
        # dm_d = self.pool.apply_async(get_distance_matrix, args=(depth_coordinates, depth_sinmod))
        t2 = time.time()
        print("Multiprocess takes: ", t2 - t1)

        t1 = time.time()
        # self.DistanceMatrix_x = dm_x.get()
        self.DistanceMatrix_x = get_distance_matrix(x_coordinates, x_sinmod)
        # self.DistanceMatrix_x = x_coordinates @ np.ones([1, len(x_sinmod)]) - np.ones([len(x_coordinates), 1]) @ x_sinmod.T
        t2 = time.time()
        print("Distance matrix - x finished, time consumed: ", t2 - t1)
        t1 = time.time()
        # self.DistanceMatrix_y = dm_y.get()
        self.DistanceMatrix_y = get_distance_matrix(y_coordinates, y_sinmod)
        # self.DistanceMatrix_y = y_coordinates @ np.ones([1, len(y_sinmod)]) - np.ones([len(y_coordinates), 1]) @ y_sinmod.T
        t2 = time.time()
        print("Distance matrix - y finished, time consumed: ", t2 - t1)
        t1 = time.time()
        # self.DistanceMatrix_depth = dm_d.get()
        self.DistanceMatrix_depth = get_distance_matrix(depth_coordinates, depth_sinmod)
        # self.DistanceMatrix_depth = depth_coordinates @ np.ones([1, len(depth_sinmod)]) - np.ones([len(depth_coordinates), 1]) @ depth_sinmod.T
        t2 = time.time()
        print("Distance matrix - depth finished, time consumed: ", t2 - t1)
        t1 = time.time()
        self.DistanceMatrix = self.DistanceMatrix_x ** 2 + self.DistanceMatrix_y ** 2 + self.DistanceMatrix_depth ** 2
        t2 = time.time()
        print("Distance matrix - total finished, time consumed: ", t2 - t1)
        t1 = time.time()
        self.ind_interpolated = np.argmin(self.DistanceMatrix, axis = 1) # interpolated vectorised indices
        t2 = time.time()
        print("Interpolation finished, time consumed: ", t2 - t1)
        self.salinity_interpolated = vectorise(salinity_sinmod[self.ind_interpolated])
        self.dataset_interpolated = pd.DataFrame(np.hstack((coordinates, self.salinity_interpolated)), columns = ["lat", "lon", "depth", "salinity"])
        t2 = time.time()
        if not filename:
            filename = fd.asksaveasfilename()
        else:
            filename = filename
        self.dataset_interpolated.to_csv(filename, index=False)
        te = time.time()
        print("Data is interpolated successfully! Time consumed: ", te - ts)

