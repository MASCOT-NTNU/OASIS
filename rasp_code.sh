diskutil list
sudo diskutil eraseDisk FAT32 SDCARD MBRFormat /dev/disk3
hostname -i
hostname -I
hostname
ip r | grep default
sudo nano /etc/resolv.conf

# == set up static IP
sudo nano /etc/dhcpcd.conf

# config
sudo raspi-config

# python3
pip3 list -v
sudo python3 setup.py install
pip3 install --upgrade pip setuptools wheel
