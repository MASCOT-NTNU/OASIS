

class GradientDetection:
    def __init__(self, 
                salinity,
                window = 201,
                salinity_max = 35,
                salinity_min = 25,
                min_event_length = 5,
                min_gradient = 0,
                min_change = 0,
                threshold_mode = "quantile_average"
                ):
        self.salinity = salinity

        self.window = window
        self.salinity_average = moving_average(self, self.salinity, self.window)
        self.diff, self.change_direction = detect_change(self)
        self.consecutive_change = get_consecutive_change(self)
        self.events = find_event(self)
        calculate_event_statistics(self)


        self.max_salinity = salinity_max
        self.min_salinity = salinity_min
        self.min_gradient = min_gradient
        self.min_event_length = min_event_length
        self.min_change = min_change
        self.threshold_mode = threshold_mode
    
        # Filtering the events
        self.positive_events = remove_events(self,dirr=[1])
        self.negative_events = remove_events(self,dirr=[-1])

        self.positive_events_joined = join_treshold(self, self.positive_events, mode = self.threshold_mode)
        self.negative_events_joined = join_treshold(self, self.negative_events, mode = self.threshold_mode)
    
        self.threshold = get_optimal_threshold(self, self.positive_events_joined, self.negative_events_joined)

def set_max_salinity(self, max_salinity):
    self.max_salinity = max_salinity

def set_min_salinity(self, min_salinity):
    self.min_salinity = min_salinity


# Some issues wher w/2 < len(x) < w
def moving_average(self, x, w, add_padding = True):
    # w should be an odd number: 
    
    # Take in measurments x, and make a rolling average with window w
    
    # This adds padding before the firsmt value
    if add_padding:
        # This is the case if the length of x is less than the windonw
        
        padd_length = round((w-1)/2)        
        
        if len(x) <= padd_length + 1:
            return np.sum(x) / (len(x)) * np.ones(len(x))
        
        else:
            
            padd_before = np.zeros(padd_length)
            padd_after = np.zeros(padd_length)
            x_flip = np.flip(x)
            for i in range(padd_length):
                padd_before[i] = np.sum(x[0:(i+padd_length + 1)] )/ (i + padd_length + 1)
                padd_after[i] = np.sum(x_flip[0:(i+padd_length + 1)] )/ (i + padd_length + 1)
            averaged_values = np.convolve(x, np.ones(w), 'valid') / w
            averaged_values = np.concatenate((padd_before, averaged_values, np.flip(padd_after)))
            return averaged_values 
    
    # this returns only averaged values
    # this list will be n - w long
    return np.convolve(x, np.ones(w), 'valid') / w



def detect_change(self):
    # This function detects change and assigns a positive or negative value for the 
    # direction of the change, and returns the actual change
    n = len(self.salinity_average)
    
    change_direction = np.ones(n-1)
    diff = np.diff(self.salinity_average)
    positive_ind = np.where(diff < 0)
    change_direction[positive_ind] = -1
    return diff, change_direction


def get_consecutive_change(self):
    # This function finds cosecutive increasing or decreasing values
    # Takes in               returns:
    # [1,1,1,1,-1,-1,1,1] ->  [1,2,3,4,-1,-2,1,2]
    
    n = len(self.change_direction)
    event_length = np.zeros(n)
    if self.change_direction[0] > 0:
        event_length[0] = 1
    else:
        event_length[0] = -1
    
    for i in range(1, n):
        if self.change_direction[i] > 0 and self.change_direction[i-1] > 0 :
            event_length[i] = event_length[i-1] + 1
        elif self.change_direction[i] < 0 and self.change_direction[i-1] < 0 :
            event_length[i] = event_length[i-1] - 1
        elif self.change_direction[i] > 0 and self.change_direction[i-1] < 0 :
            event_length[i] = 1
        else:
            event_length[i] = -1
    return event_length



def find_event(self):
    # This function makes an event dictionary
    # An event is consecutive increasing or decreasing salinity
    # eks: {"ind_start": 10, "ind_end":15, "length":5, "dir":1}
            
    events = []
    event = {"ind_start":0, "ind_end":0, "length":0, "dir":self.consecutive_change[0]}
    for i, counter in enumerate(self.consecutive_change):
       
        if i == len(self.consecutive_change) - 1:
           
            event["ind_end"] = i
            event["length"] = event["ind_end"]- event["ind_start"] + 1
            if counter < 0:
                event["dir"] = -1
            else:
                event["dir"] = 1

            events.append(event)
            break
        
        if self.consecutive_change[i] > 0 and self.consecutive_change[i-1] < 0 and i != 0:
            
            event["ind_end"] = i-1
            event["length"] = event["ind_end"]- event["ind_start"] + 1
            event["dir"] = -1
       
            events.append(event)
            event = {"ind_start":i, "ind_end":0, "length":0}
            
        if self.consecutive_change[i] < 0 and self.consecutive_change[i-1] > 0 and i != 0:
            event["ind_end"] = i-1
            event["length"] = event["ind_end"]- event["ind_start"] + 1
            event["dir"] = 1
            
            events.append(event)
            event = {"ind_start":i, "ind_end":0, "length":0}      
    return events


                    
                

def calculate_event_statistics(self, mode = "average"):
    # Here we calculate relevant statistics for the event
    
    # The event is defined like this:
    # event = {
    #       "ind_start": startin index of the event
    #       "ind_end": ending index of the event
    #       "length": length of the event
    #       "dirr": 1 for increasing and -1 for decreascing
    #       "max": max salinity of the event
    #       "min": minimum salinity of the event
    #       "dfff": list of the diferences
    #       "change": overall change
    #       "avg_gradient": change / length
    #       "max_gradient": max(diff)
    #       "salinity": salinity values
    #       "threshold": the threshold for the event

    for event in self.events:
        ind_start, ind_end = event["ind_start"], event["ind_end"]
        salinity_event = np.array([salinity[event["ind_start"]]])
        if event["length"] > 1:
            salinity_event = self.salinity_average[ind_start: ind_end]
        event["salinity"] = salinity_event
        event["diff"] = np.diff(salinity_event)
        event["max"] = max(salinity_event)
        event["min"] = min(salinity_event)
        event["change"] = (event["max"] - event["min"]) * event["dir"]
        event["avg_gradient"] = event["change"] / event["length"]
        event["max_gradient"] = max(abs(np.diff(salinity)))
        if mode == "average":
            event["threshold"] = np.mean(salinity_event)
        if mode == "quantile":
            salinity_sorted = np.sort(salinity_event)
            quantile = 0.8
            ind = round(len(salinity_sorted) * quantile)
            event["threshold"] = salinity_sorted[ind]
            
        if mode == "quantile_mean":
            salinity_sorted = np.sort(salinity_event)
            quantile = 0.8
            ind_low = round(len(salinity_sorted) * quantile)
            ind_high = round(len(salinity_sorted) * (1- quantile))
            event["threshold"] = (salinity_sorted[ind_low] + salinity_sorted[ind_high]) / 2 


def remove_events(self, dirr=[-1,1]):
    # This removes the events that do not meet the specified criterias
    new_events = []
    for event in self.events:
        if (event["length"]>= self.min_event_length) and np.abs(event["change"]) >= self.min_change and (event["dir"] in dirr) and np.abs(event["avg_gradient"])>self.min_gradient and event["max"]>self.min_salinity and event["min"]< self.max_salinity:
            new_events.append(event)
    return new_events

def get_optimal_threshold(self, positive_events_joined, negative_events_joined):
        if positive_events_joined["comulative_change"] > negative_events_joined["comulative_change"]:
            return positive_events_joined["threshold"]
        else:
            return negative_events_joined["threshold"]

def join_treshold(self, event_dict, mode = "average"):
    
    joined_events = {
        "comulative_change": 0,
        "total_length": 0,
        "threshold": 0,
        "salinity": np.array([]), 
        "indecies": np.array([]),
        "diff": np.array([]),
        "max_diff": 0,
        "max_diff_index": 0, 
    }
    for event in event_dict:
        joined_events["comulative_change"] += np.sum(np.abs(event["diff"]))
        joined_events["salinity"] =  np.concatenate((joined_events["salinity"], event["salinity"]))
        joined_events["diff"] =  np.concatenate((joined_events["diff"],np.array([0]), event["diff"]))
        joined_events["indecies"] =  np.concatenate((joined_events["indecies"],
                                                     np.arange(event["ind_start"], event["ind_end"])))
        joined_events["total_length"] += event["length"]
    
    if joined_events["total_length"] > 0:
        if mode == "average":
            weights = np.zeros(len(event_dict))
            thresholds = np.zeros(len(event_dict))
            for i, event in enumerate(event_dict):

                weights[i] = event["length"]
                thresholds[i] = event["threshold"]
            joined_events["threshold"] = np.sum(weights * thresholds) / np.sum(weights)

        if mode == "quantile_average":

            # returns (x_q +  x_q-1)/2, wher q is the quantile
            all_salinity = np.array([])
            for i, event in enumerate(event_dict):
                all_salinity =  np.concatenate((all_salinity, event["salinity"]))
            salinity_sorted = np.sort(all_salinity)
            quantile = 0.8
            ind_low = round(len(salinity_sorted) * quantile)
            ind_high = round(len(salinity_sorted) * (1- quantile))
            joined_events["threshold"] =  (salinity_sorted[ind_low] + salinity_sorted[ind_high]) / 2 


        if mode == "longest":
            longest = 0
            longest_length = 0
            threshold = 0
            for i, event in enumerate(event_dict):
                if event["length"] > longest_length:
                    longest = i
                    longest_length = event["length"]
                    threshold = event["threshold"]
            joined_events["threshold"] =  threshold

        if mode == "steepest_gradient":
            steepest = 0
            salinity = 0
            for i, event in enumerate(event_dict):

                ind_max = np.argpartition(np.abs(event["diff"]), 1)[0]

                if np.abs(event["diff"][ind_max]) > steepest:
                    salinity = event["salinity"][ind_max]
                    steepest = np.abs(event["diff"][ind_max])
            joined_events["threshold"] =  salinity
    return joined_events


   
    
        
        
        
        
    


if __name__=="__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd 
    import time

    ## Get Data
    #data = np.load("data/transect1_raw.npz")
    #salinity = data['salinity']
    #lat = data['lat']
    #lon = data['lon']
    #depth = data['depth']

    salinity = np.array([1,2,3,4,5,6,7,4,5,2,5,3,5,6,7,8,5,3,5,7])
    a = time.time()
    threshold_detector = GradientDetection(salinity,salinity_max=100,salinity_min=0, window = 3, min_event_length = 3)



    b = time.time()
    print(b  - a)

    #print(threshold_detector.salinity_average)
    #print( threshold_detector.diff, threshold_detector.change_direction)
    print(threshold_detector.positive_events_joined)
    print(threshold_detector.negative_events_joined)
    print(threshold_detector.threshold)
    


