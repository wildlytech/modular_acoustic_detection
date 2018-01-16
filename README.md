# Modular Acoustic Detection

## Environment Setup

To download all the relevant Ubuntu packages:
```shell
# Make script executable
$ chmod 777 ubuntu_packages_install.sh

# Run script to install
$ ./ubuntu_packages_install.sh
```

## Local repo setup

To load all git submodules:
```shell
$ git submodule update --init --recursive
```

To download all the data files:
```shell
# Make script executable
$ chmod 777 download_data_files.sh

$ ./download_data_files.sh
```

To download all the sound files (This is a really lengthy process and hence is not advisable unless you absolutely have to):
```shell
$ python download_all_sounds.py
```
