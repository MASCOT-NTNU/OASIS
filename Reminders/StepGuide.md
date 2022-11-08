Neptus set up guide
First, use command `./home/lsts/workspace/neptus/develop/netpus auv` to launch neptus.
Second, click `View` on the menu and click `Plugin Manager` to find `FollowReference Interaction for NTNU` in the left column and then click `Add`, then NTNU log should appear in the left column in neptus.
Third, click on NTNU log (`FollowReference Interaction for NTNU`), and move the mouse in neptus field, and then right click anywhere on the map, click `Follow Referance Settings`, and then change `Control timeout` to be `200`.


# Plan A step-by-step guide
First, click on NTNU log (`FollowReference Interaction for NTNU`), and then right click on the map, and finally click `Activate Follow Reference for lauv-xplore-1`

# Plan B step-by-step guide
First, `screen -S mascot` in dell PC terminal.
Then, use command `ssh xp1m` log into the main CPU on lauv-xp1, passwd: `root`
Afterwards, use command `ssh lsts@10.0.10.123` to log in the backseat CPU on lauv-xp1, passwd: `root`
Finally, use `cd` to go to the home directory, in the home directory, there should exist two bash scripts starting with `planb_*.sh`. Launch the bridge with `/bin/bash planb_bridge.sh` and then the main program with `/bin/bash planb_op2.sh`.
Then you can wait a few more minutes before you activate follow reference NTNU.
After that, then you can detach the session by press keys `Ctrl+a` and then immediately key `d` so that the session keeps alive in the background. Later you can attach the session back by type `screen -r` to resume the session. 
