# Note for André

- Step I, launch the bridge `roslaunch bridge.launch`.
- Step II, launch the mission script `python3 Launcher.py` in each src folder, OP1 and OP2. 

- launch everthing with `> /dev/null 2>&1 &`


# Commands that are useful
- `htop`, to see the processes running in the background.
- `ip r`, to show the ip address of the client.
- `arp -a`, to see all visible clients.
- `ping 10.0.10.123`, to ping the backseat CPU on LAUV-XP1.
- `ping 10.0.10.120`, to ping the main CPU on LAUV-XP1.
- `ssh lsts@10.0.10.123`, to log in to the terminal in the AUV, passwd: `root`.
