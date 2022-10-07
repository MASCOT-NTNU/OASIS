#!/usr/bin/env python3
# license removed for brevity
# Adaptive sampling group of NTNU
# Tore Mo-Bjørkelund 2021
# contact: tore.mo-bjorkelund@ntnu.no

import rospy
import numpy as np
from auv_handler import AuvHandler
import imc_ros_interface
from imc_ros_interface.msg import Temperature, Salinity, EstimatedState, Sms
from math import radians
import time

"""
This agenent will move in a rectangle
At the third waypoint it will send an SMS message 
It will sample the temperature and the salinity continously

TODO: add a popup function
      add some relevant computations
"""


class HarbourExample:
    def __init__(self):
        self.node_name = 'Harbour_test'
        rospy.init_node(self.node_name,anonymous=True)
        self.rate = rospy.Rate(1) # 1Hz
        self.auv_handler = AuvHandler(self.node_name,"Harbour_test")

        rospy.Subscriber("/Vehicle/Out/Temperature_filtered", Temperature, self.TemperatureCB)
        rospy.Subscriber("/Vehicle/Out/Salinity_filtered", Salinity, self.SalinityCB)
        rospy.Subscriber("/Vehicle/Out/EstimatedState_filtered", EstimatedState, self.EstimatedStateCB)


        self.speed = 1.6 #m/s
        self.depth = 0.0 #meters
        self.lat = 41.185881*np.pi/180.0
        self.lon = -8.706053*np.pi/180.0
        self.last_state = "unavailable"
        self.rate.sleep()

        # Phone number for the SMS
        self.phone_number = "+4440040327"

        self.auv_handler.setWaypoint(self.lat, self.lon, self.depth)
        self.init = True
        self.currentTemperature = 0.0
        self.currentSalinity = 0.0

        # Pre-scripted waypoints for the AUV to move in 
        self.waypoints = [[-8.706053,41.185881, 0],
                        [-8.706805,41.185501, 0.5],
                        [-8.706172,41.184888, 0],
                        [-8.705431,41.185291, 0],
                        [-8.706053,41.185881, 0]]

        self.counter = 0
        self.max_count = len(self.waypoints)

        # The dataset contains: lat, lon, depth, salinity, temperature, time
        self.dataset = np.empty([0, 6])
       
        self.temperature = []
        self.salinity = []



    def TemperatureCB(self,msg):
        self.currentTemperature = msg.value.data
    
    def SalinityCB(self, msg):
        self.currentSalinity = msg.value.data

    def EstimatedStateCB(self, msg):
        self.lat = msg.lat.data 
        self.lon = msg.lon.data
        self.depth = msg.depth.data


    def send_sms(self):
        SMS = Sms()
        SMS = Sms()
        SMS.number.data = self.phone_number
        SMS.timeout.data = 60
        SMS.contents.data = "Congrats, sms sucsess!"
        self.sms_pub_.publish(SMS)
        print("Finished SMS sending!")




    def run(self):
        while not rospy.is_shutdown():
            if self.init:

                lat = self.lat
                lon = self.lon
                depth = self.depth
                temperature = self.currentTemperature
                time = time.time()
                salinity = self.currentSalinity
                
                self.dataset = np.append(self.dataset, 
                        np.array([lat, lon, depth, salinity, temperature, time]).reshape(1, -1), axis=0)
                if self.auv_handler.getState() == "waiting":
                    lat = self.waypoints[self.counter][0]*np.pi/180.0
                    lon = self.waypoints[self.counter][1]*np.pi/180.0
                    depth = self.waypoints[self.counter][2]
                    self.auv_handler.setWaypoint(lat, lon, depth = depth)
                    self.counter += 1

                    if self.counter == 2: 
                        #send message
                        self.auv_handler.sendSMS()
                    print(self.counter)

                self.last_state = self.auv_handler.getState()
                self.auv_handler.spin()
            self.rate.sleep()


if __name__ == "__main__":
    gpex = HarbourExample()
    try:
        gpex.run()
    except rospy.ROSInterruptException:
        pass