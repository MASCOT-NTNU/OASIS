# Scenario analysis

Scenario I: bridge is killed, OP is alive.
- check if OP is still running.
- relaunch the bridge `/bin/bash bridgelauncher.sh`

Scenario II: both bridge and OP are killed.
- reboot the AUV or just the backseat CPU.

Scenario III: OP is killed, bridge is alive.
- possibility I: OP is complete
- possibility II: crash, relaunch

Scenario IV: both bridge and OP are alive.
- Then everything is good, just need to reactivate the operations.

Scenario V: AUV reboot.
- Just need to reactivate the operations. 
