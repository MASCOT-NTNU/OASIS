"""
Harbour test will conduct the test in the Porto harbour given a certain behaviour and waypoints.
It needs to test all possible states from adaframe.
- It should be able to move to next waypoint.
- It should be able to send SMS when required.
- It should be able to popup when time is up.
- It should be able to update the field given a certain model.
- It should be able to resume the adaptive behaviour when mission is possibly aborted.
"""
import os

import rospy
import numpy as np
from auv_handler import AuvHandler
import imc_ros_interface
from imc_ros_interface.msg import Temperature, Salinity, EstimatedState, Sms
import math
import time
import pathlib
import pandas as pd


def checkfolder(folder):
    path = pathlib.Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    # print(folder + "is created")


class HarbourTest:
    def __init__(self):
        self.node_name = 'Harbour_test'
        rospy.init_node(self.node_name,anonymous=True)
        self.rate = rospy.Rate(1)  # 1Hz
        self.auv_handler = AuvHandler(self.node_name,"Harbour_test")

        rospy.Subscriber("/Vehicle/Out/Temperature_filtered", Temperature, self.TemperatureCB)
        rospy.Subscriber("/Vehicle/Out/Salinity_filtered", Salinity, self.SalinityCB)
        rospy.Subscriber("/Vehicle/Out/EstimatedState_filtered", EstimatedState, self.EstimatedStateCB)

        self.speed = 1.5  # [m/s]
        self.depth = 0.0  # [m]
        self.lat = math.radians(41.185881)
        self.lon = math.radians(-8.706053)
        self.last_state = "unavailable"
        self.rate.sleep()

        """ POP UP handler. """
        self.min_surface_time = 90  # [sec]
        self.max_submerged_time = 120  # [sec]

        """ SMS handler. """
        self.__phone_number = "+4740040327"
        self.__iridium_destination = "manta-3"
        self.sms_pub_ = rospy.Publisher("/IMC/In/Sms", Sms, queue_size=10)

        """ Set the waypoint """
        self.auv_handler.setWaypoint(self.lat, self.lon, self.depth)
        self.init = True
        self.currentTemperature = 0.0
        self.currentSalinity = 0.0

        """ File handler. """
        t = int(time.time())
        self.foldername = os.getcwd() + "/raw_ctd/{:d}/".format(t)
        checkfolder(self.foldername)

        # Pre-scripted waypoints for the AUV to move in
        # This will run for three "loops"
        self.waypoints = np.array([[41.185881, -8.706053, 0],
                                   [41.185501, -8.706805, 0.5],
                                   [41.184888, -8.706172, 0],
                                   [41.185291, -8.705431, 0],
                                   [41.185881, -8.706053, 0],
                                   [41.185501, -8.706805, 0.5],
                                   [41.184888, -8.706172, 0],
                                   [41.185291, -8.705431, 0],
                                   [41.185881, -8.706053, 0]
                                   [41.185501, -8.706805, 0.5],
                                   [41.184888, -8.706172, 0],
                                   [41.185291, -8.705431, 0],
                                   [41.185881, -8.706053, 0]])

        self.counter = 0
        self.max_count = len(self.waypoints)

        """ 
        Dataset that saves the mission data from the harbour test.
        :param
            timestamp: timestamp in integer.  
            lat: distance along north direction. 
            lon: distance along east direction. 
            depth: distance along depth direction.
            salinity: salinity value from CTD sensor in ppt, [particle per thousand]
            temperature: temperature value from CTD sensor in degrees Celcius.
        """
        self.dataset = np.empty([0, 6])
       
        self.temperature = []
        self.salinity = []

    def TemperatureCB(self, msg) -> None:
        """ Refresh temperature data from ROS message. """
        self.currentTemperature = msg.value.data
    
    def SalinityCB(self, msg) -> None:
        """ Refresh salinity data from ROS message. """
        self.currentSalinity = msg.value.data

    def EstimatedStateCB(self, msg) -> None:
        """ Refresh current estimated location data from ROS message. """
        self.lat = msg.lat.data 
        self.lon = msg.lon.data
        self.depth = msg.depth.data

    def send_SMS_mission_complete(self):
        print("Mission complete! will be sent via SMS")
        SMS = Sms()
        SMS.number.data = self.__phone_number
        SMS.timeout.data = 90
        SMS.contents.data = "Congrats, Mission complete! Lets go to the sea!"
        self.sms_pub_.publish(SMS)
        print("Finished SMS sending!")

    def send_SMS(self, sms_string):
        SMS = Sms()
        SMS.number.data = self.__phone_number
        SMS.timeout.data = 90
        SMS.contents.data = sms_string
        self.sms_pub_.publish(SMS)

    def run(self):
        t_last_popup = time.time()
        while not rospy.is_shutdown():
            if self.init:
                self.dataset = np.append(self.dataset,
                                         np.array([time.time(), self.lat, self.lon, self.depth,
                                                   self.currentSalinity, self.currentTemperature]).reshape(1, -1),
                                         axis=0)
                if self.auv_handler.getState() == "waiting":
                    t_popup = time.time()
                    if t_popup - t_last_popup > self.max_submerged_time:
                        self.auv_handler.PopUp(sms=True, iridium=True, popup_duration=self.min_surface_time,
                                               phone_number=self.__phone_number,
                                               iridium_dest=self.__iridium_destination)
                        t_last_popup = time.time()

                    wp = self.waypoints[self.counter]
                    self.auv_handler.setWaypoint(math.radians(wp[0]), math.radians(wp[1]), wp[2])
                    self.counter += 1

                    """ Now it is time to save data. """
                    df = pd.DataFrame(self.dataset, columns=['timestamp', 'lat', 'lon', 'depth', 'salinity', 'temperature'])
                    df.to_csv(self.foldername + "D_{:03d}.csv".format(self.counter))
                    self.dataset = np.empty([0, 6])

                    if self.counter >= self.max_count:
                        self.send_SMS_mission_complete()
                        rospy.signal_shutdown("Mission complete! ")
                        break
                    print(self.counter)

                self.last_state = self.auv_handler.getState()
                self.auv_handler.spin()
            self.rate.sleep()


if __name__ == "__main__":
    h = HarbourTest()
    h.run()

