import numpy as np
import scipy.io.wavfile as wavfile
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
from sonifyfn import sonify

# the sounds i made have sample rate 48,000 Hz and they are 5 seconds long
# the original files are 2MHz sample rate and 1 second long.
# did i convert from 2MHz to 240KHz and then play like they are 48KHz?

# ya got me.
A = sio.loadmat('cyberpowerups_001.mat')
B = sio.loadmat('cortelcophone_001.mat')
sounds = [None] * 3
N = 7

for ik in range(3):
    if ik == 0:
        TheData = A['data'].astype(float)
        sOut = 'cyberpower.wav'
    elif ik == 1:
        TheData = B['data'].astype(float)
        sOut = 'cortelcophone.wav'
    else:
        TheData = 0.5 * (A['data'].astype(float) + B['data'].astype(float))
        sOut = 'combined.wav'
    fs = A['samp_rate'].item()
    fs = float(fs)
    pOut = sOut[:-4] + '.png'
    sonify(TheData, fs, sOut, pOut, N=N)