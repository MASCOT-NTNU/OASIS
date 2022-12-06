# Checklist
## What to do before launching the mission
- clean data folder in GMRF folder.
- change SMS phone number to Choi's phone.
- remove data folders and contents from grf modules in long-horizon or gmrf.
- check all parameters in config to make it compatible with mission operation condition "wind_dir, wind_level, clock_start, clock_end".
- check set waypoint, depth needs to be positive in adaframe.
- activate all numbers in neptus in system configuration
- change follow reference timeout in follow reference setting.
- change ip address
- launch bridge and then script.

- check all the scripts are launch in the no-hop mode.


# Change during mission
- change `threshold.txt` and reboot the AUV. 
- change threshold in both op1 and op2.
- change waypoint number after the first run to 30/50.
- [x] change iridium destination to the manta we use in the operation.
