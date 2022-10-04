Usr CPU: root@10.0.10.150
passwd: root

Backseat CPU: 10.0.10.153
passwd: root

XP1
main CPU: 10.0.10.120
backseat CPU: 10.0.10.123
Usr: lsts
passwd: root

Self IP: 10.0.248.61
Subnet Mask: 255.255.0.0
Router: 10.0.0.1
DNS: 8.8.8.8


#TODO:
- remember to add `id_rsa.pub` to `authorized.keys`

- Step I: copy [Transports.TCP.Server/Backseat] to the main CPU `lauv-xplore-1.ini`
- Step II: Change IP address at IMC_ROS_launch.bridge and check port number.

- WARNING: Launch all the commands in the backseat CPU with `> /dev/null 2>&1 &`.
- check process ID: pidof python3/python

- check if AUV has HITL mode.

- SPDE remember to tell Martin about the wind condition one day ahead of mission.
