"""
Different audio codecs are used to measure the latency
"""
import time
import subprocess

START = time.time()
subprocess.call('ffmpeg -i speech.wav  -c:a aac -b:a 64k  speech.aac', shell=True)
print 'Timing for aac :', (time.time() - START)
START1 = time.time()
subprocess.call('ffmpeg -i speech.wav  -c:a ac3 -b:a 64k  speech.ac3', shell=True)
print 'Timing for ac3 :', (time.time() -START1)
START2 = time.time()
subprocess.call('ffmpeg -i speech.wav  -c:a mp2 -b:a 64k  speech.mp2', shell=True)
print 'Timing for mp2 :', (time.time() -START2)
START3 = time.time()
subprocess.call('ffmpeg -i speech.wav  -c:a flac -b:a 64k  speech.flac', shell=True)
print 'Timing for flac :', (time.time() - START3)
START4 = time.time()
subprocess.call('ffmpeg -i speech.wav  -c:a libopus -b:a 64k  speech.opus', shell=True)
print 'Timing for libopus :', (time.time() - START4)
START5 = time.time()
subprocess.call('ffmpeg -i speech.wav  -c:a mp3 -b:a 64k  speech.mp3', shell=True)
print 'Timing for mp3 :', (time.time() - START5)
