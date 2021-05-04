#!/bin/sh

# Get python3.7
apt-get install python3.7 python3.7-dev python3.7-distutils python3.7-gdbm python3.7-lib2to3 python3.7-minimal python3.7-tk python3-setuptools python3-pip

# Set default python to python3 (if not already) and default python3 to python 3.7
# sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
# sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 2

# Another way to alias python3 but dependencies don't line up
# echo '' >> ~/.bashrc
# echo 'alias python3=/usr/bin/python3.7' >> ~/.bashrc
# echo 'alias python=python3' >> ~/.bashrc
# echo '' >> ~/.bashrc

# Minimum needed for loading requirements file is pip and virtualenv
pip install --upgrade pip
pip install virtualenv

# Other libraries
apt-get install ffmpeg
