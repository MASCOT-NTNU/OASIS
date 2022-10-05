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

# Packages installation
pip3 install numpy
pip3 install scipy
pip3 install shapely
# install libsuitesparse-dev before sksparse
sudo apt-get install -y libsuitesparse-dev
pip3 install scikit-sparse
pip3 install pandas
pip3 install pyyaml
pip3 install rospkg
pip3 install defusedxml
pip3 install netifaces
