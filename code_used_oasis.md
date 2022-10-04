# Here comes common codes for OASIS operation.

---
## Copy figures from MOHID to local laptop to observe.

`scp -r gpu03:/home/yaoling/OASIS/fig/mohid/* /Users/yaolin/OneDrive\ -\ NTNU/MASCOT_PhD/Projects/OASIS/fig/MOHID/.`

---

---
## Making animations: To change frame rate and vide filename

`ffmpeg -r 15 -i P_%03d.png -vcodec libx264 -crf 20 -pix_fmt yuv420p mohid.mp4`

---
