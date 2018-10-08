import numpy as np
import pandas as pd
import pickle
import pydub
from pydub import AudioSegment
import time
import subprocess

start=time.time()
subprocess.call('ffmpeg -i speech.wav  -c:a aac -b:a 64k  speech.aac', shell=True)
print 'Timing for aac :', (time.time() - start)
start1=time.time()
subprocess.call('ffmpeg -i speech.wav  -c:a ac3 -b:a 64k  speech.ac3', shell=True)
print 'Timing for ac3 :', (time.time() -start1)
start2=time.time()
subprocess.call('ffmpeg -i speech.wav  -c:a mp2 -b:a 64k  speech.mp2', shell=True)
print 'Timing for mp2 :', (time.time() -start2)
start3=time.time()
subprocess.call('ffmpeg -i speech.wav  -c:a flac -b:a 64k  speech.flac', shell=True)
print 'Timing for flac :', (time.time() - start3)
start4=time.time()
subprocess.call('ffmpeg -i speech.wav  -c:a libopus -b:a 64k  speech.opus', shell=True)
print 'Timing for libopus :', (time.time() - start4)
start5=time.time()
subprocess.call('ffmpeg -i speech.wav  -c:a mp3 -b:a 64k  speech.mp3', shell=True)
print 'Timing for mp3 :', (time.time() - start5)
