#!/bin/sh

# Get python3.8
apt-get install python3.8 python3.8-dev python3.8-distutils python3.8-gdbm python3.8-lib2to3 python3.8-minimal python3.8-tk python3-setuptools python3-pip

# Set default python to python3 (if not already) and default python3 to python 3.8
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 2

# Another way to alias python3 but dependencies don't line up
# echo '' >> ~/.bashrc
# echo 'alias python3=/usr/bin/python3.8' >> ~/.bashrc
# echo 'alias python=python3' >> ~/.bashrc
# echo '' >> ~/.bashrc

# Minimum needed for loading requirements file is pip and virtualenv
pip install --upgrade pip
pip install virtualenv

# Other libraries
apt-get install ffmpeg
