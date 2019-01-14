"""
Different audio codecs are used to measure the latency.
Timings for each type of audio codecs will be displayed onto terminal with green color.
"""
import time
import subprocess
from colorama import Fore, Style

START = time.time()
subprocess.call('ffmpeg -i speech.wav  -c:a aac -b:a 64k  speech.aac', shell=True)
print Fore.GREEN + '\n\nTiming for aac :', (time.time() - START)
print Style.RESET_ALL
START1 = time.time()
subprocess.call('ffmpeg -i speech.wav  -c:a ac3 -b:a 64k  speech.ac3', shell=True)
print Fore.GREEN +'\n\nTiming for ac3 :', (time.time() -START1)
print Style.RESET_ALL
START2 = time.time()
subprocess.call('ffmpeg -i speech.wav  -c:a mp2 -b:a 64k  speech.mp2', shell=True)
print Fore.GREEN +'\n\nTiming for mp2 :', (time.time() -START2)
print Style.RESET_ALL
START3 = time.time()
subprocess.call('ffmpeg -i speech.wav  -c:a flac -b:a 64k  speech.flac', shell=True)
print Fore.GREEN +'\n\nTiming for flac :', (time.time() - START3)
print Style.RESET_ALL
START4 = time.time()
subprocess.call('ffmpeg -i speech.wav  -c:a libopus -b:a 64k  speech.opus', shell=True)
print Fore.GREEN +'\n\nTiming for libopus :', (time.time() - START4)
START5 = time.time()
print Style.RESET_ALL
subprocess.call('ffmpeg -i speech.wav  -c:a mp3 -b:a 64k  speech.mp3', shell=True)
print Fore.GREEN +'\n\nTiming for mp3 :', (time.time() - START5)
print Style.RESET_ALL
