# Here comes common codes for OASIS operation.

---
## Here comes code to copy src to AUV.

`scp -r * xp1:/home/lsts/catkin_ws/src/adaframe_examples/OP2/src/.`
`scp -r * lauv-xp1:/home/lsts/catkin_ws/src/adaframe_examples/OP1_OCT11/src/.`

---

---
## Copy figures from MOHID to local laptop to observe.

`scp -r gpu03:/home/yaoling/OASIS/fig/mohid/* /Users/yaolin/OneDrive\ -\ NTNU/MASCOT_PhD/Projects/OASIS/fig/MOHID/.`

---

---
## Copy GPU server to local machine.
`scp -r gpu03:/home/yaoling/OASIS/OP2/ .`

---

---
## Making animations: To change frame rate and vide filename

`ffmpeg -r 15 -i P_%03d.png -vcodec libx264 -crf 20 -pix_fmt yuv420p mohid.mp4`

---

---
## Remove large files from git history
`git filter-branch --force --index-filter "git rm --cached --ignore-unmatch OP2_LongHorizon/src/Planner/RRTSCV/RRT_Random_Locations.npy" --prune-empty --tag-name-filter cat -- --all`
---
