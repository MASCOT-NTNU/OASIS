from concurrent.futures import wait
from re import S
from turtle import update
from numpy import empty, quantile
import numpy as np

class BoundaryFinder:

    def __init__(self, threshold = 0):
        self.threshold = threshold

        # In-situ measurments
        self.salinity = np.array([])
        self.depth = np.array([])
        self.x = np.array([])
        self.y = np.array([])
        
        depth_target = 0.5
        delta_depth_bound = 0.25
        self.depth_boundary = np.array([depth_target - delta_depth_bound, depth_target + delta_depth_bound])



    def find_threshold_location(self):

        if self.threshold != 0:
        
            indecies = np.where(np.logical_and(self.salinity_average < self.threshold + 1, self.salinity_average > self.threshold - 1), True, False)

            current_x = self.x[-1]
            current_y = self.y[-1]

            x_values = self.x[indecies]
            y_values = self.y[indecies]

            x_dist = (current_x - x_values)**2
            y_dist = (current_y - y_values)**2
            dist = x_dist + y_dist
            min_ind = np.argmin(dist)

            return x_values[min_ind], y_values[min_ind]
        return 0,0



   

    

    def update_measurments(self, salinity, depth, x, y):

        # Sets the four main variables
        self.x = x
        self.y = y
        self.salinity = salinity
        self.depth = depth

        # Cleans avay the measurments that are not in the depth bound defined
        filter_measurments(self)

        if len(self.salinity) > self.window:

            # Update the rolling average
            self.salinity_average = moving_average(self, self.salinity, self.window)
            self.depth_average = moving_average(self, self.depth, self.window)
    
    def filter_measurments(self):
        indecies = np.logical_and(self.depth < self.depth_boundary[1], self.depth > self.depth_boundary[0])
        self.depth = self.depth[indecies]
        self.salinity = self.salinity[indecies]
        self.x = self.x[indecies]
        self.y = self.y[indecies]
        

    def set_threshold(self, threshold):
        self.threshold = threshold

        pass


    def is_threshold_found(self):
        # Returns True/False based on if it belives it has found the threshold
        # Want several measurments above the threshold and below the threshold
        salinity_above_threshold = self.salinity_average[self.salinity_average > self.threshold]
        salinity_below_threshold = self.salinity_average[self.salinity_average < self.threshold]
        if len(salinity_above_threshold) > 20 and len(salinity_below_threshold) > 20:
            return True
        return False


    def moving_average(self, x, w, add_padding = True):
        # w should be an odd number:

        # Take in measurments x, and make a rolling average with window w

        # This adds padding before the firsmt value

        if len(x) == 0:
            return np.empty((0))
        if add_padding:
            # This is the case if the length of x is less than the windonw

            padd_length = round(( w -1 ) /2)

            if len(x) <= padd_length + 1:
                return np.sum(x) / (len(x)) * np.ones(len(x))

            else:

                padd_before = np.zeros(padd_length)
                padd_after = np.zeros(padd_length)
                x_flip = np.flip(x)
                for i in range(padd_length):
                    padd_before[i] = np.sum(x[0:( i +padd_length + 1)] )/ (i + padd_length + 1)
                    padd_after[i] = np.sum(x_flip[0:(i + padd_length + 1)]) / (i + padd_length + 1)
                averaged_values = np.convolve(x, np.ones(w), 'valid') / w
                averaged_values = np.concatenate((padd_before, averaged_values, np.flip(padd_after)))
                return averaged_values

                # this returns only averaged values
        # this list will be n - w long
        return np.convolve(x, np.ones(w), 'valid') / w


if __name__ == "__main__":
    ## Get Data
    print("jfdsf")
    path = "/Users/ajolaise/OneDrive - NTNU/PhD/AUV Missions/Porto October 2022/code/data"
    data = np.load(path + "/transect1_raw.npz")
    salinity = data['salinity']
    depth = data['depth']
    x = data['lat']
    y = data['lon']
    test_case = 1

    if test_case == 1:

        print("Case ", test_case)
        a = time.time()
        threshold_finder = BoundaryFinder(threshold=24)
    
        for i in range(240):
            k = 0
            m = i * 100
            if m > len(salinity) -1:
                break
                m = len(salinity) - 1

            update_measurments(threshold_finder, salinity[k:m], depth[k:m], x[k:m], y[k:m])
            if threshold_found(threshold_finder):
                #print(find_threshold_location(threshold_detector))
                pass
            print(threshold_found(threshold_detector))
            print(threshold_detector.threshold)