import numpy as np
import scipy.io.wavfile as wavfile
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
from sonifyfn import sonify, create_directories

# the sounds i made have sample rate 48,000 Hz and they are 5 seconds long
# the original files are 2MHz sample rate and 1 second long.
# did i convert from 2MHz to 240KHz and then play like they are 48KHz?

# ya got me.

directory = 'E:/FlamingMoeByDevice'
N = 5

for root, dirs, files in os.walk(directory):
    if "SEPTA" in dirs:
        dirs.remove("SEPTA")  # Skip the "SEPTA" directory and its contents
        print("Skipped directory: SEPTA")

    for file in files:

        filepath = os.path.join(root, file).replace('\\', '/')
        print(filepath)
        if not file.endswith('.mat'):
            continue
        sample = sio.loadmat(filepath)
        try:
            TheData = sample['data'].astype(float)
            fs = sample['samp_rate'].item()
        except Exception as e:
            print(f'Error: {e}. Continuing...')
            continue
        fs = float(fs)
        # Basically the absolute path of the file (minus the drive name) adds that to your current directory. Makes all dirs needed
        sOut = create_directories(filepath[3:])
        pOut = os.path.join(os.path.dirname(sOut), file[:-4] + '.png')
        sonify(TheData, fs, sOut, pOut, N=N)
        #Sounds array update?