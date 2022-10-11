# Note for André

- Step 0, whenever you are in a folder called `catkin_ws`, please set up the environment using `source devel/setup.bash`. 
- Step I, launch the bridge `roslaunch bridge.launch` to check if everything is running correctly.
- If so, launch the bridge in no-hop mode `roslaunch bridge.launch > /dev/null 2>&1 &`
- Step II, launch the mission script `python3 Launcher.py` in each src folder, OP1 and OP2 first to check if everything is running correctly.
- If so, launch the mission script in no-hop mode `python3 Launcher.py > /dev/null 2>&1 &`
- Step III, check `htop` to obverse if all the processes are running properly.
- If so, wait maybe roughly 30 seconds before activating the program in neptus.

- launch everthing with `> /dev/null 2>&1 &`


# Commands that are useful
- `htop`, to see the processes running in the background.
- `ip r`, to show the ip address of the client.
- `arp -a`, to see all visible clients.
- `ping 10.0.10.123`, to ping the backseat CPU on LAUV-XP1.
- `ping 10.0.10.120`, to ping the main CPU on LAUV-XP1.
- `ssh lsts@10.0.10.123`, to log in to the terminal in the AUV, passwd: `root`.
