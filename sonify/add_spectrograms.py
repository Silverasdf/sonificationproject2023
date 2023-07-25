# Add spectrograms - Takes two files from directory of .mat files, adds them together, and makes the spectrograms
# Ryan Peruski, 07/24/23
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
from sonifyfn import sonify

data_dir = '/root/SonificationProject/Data/SomeMatFiles'
picture_out = '/root/SonificationProject/Data/sonified_spectrograms_off'
sound_out = '/root/SonificationProject/Data/sonified_spectrograms_off'
num_files = 100

#Choose random file from directory
eligible_files = [file for file in os.listdir(data_dir) if file.endswith('2.mat')]

num = 0
for file in eligible_files:
    for file2 in eligible_files:
        num += 1
        print(f'Processing file {num} of {len(eligible_files)**2}')
        filepath = os.path.join(data_dir, file).replace('\\', '/')
        filepath2 = os.path.join(data_dir, file2).replace('\\', '/')

        sample = sio.loadmat(filepath)
        sample2 = sio.loadmat(filepath2)
        TheData = []
        data1 = sample['data'].astype(float)
        data2 = sample2['data'].astype(float)
        for i in range(len(data1)):
            TheData.append((data1[i] + data2[i]))
        fs = sample['samp_rate'].item()
        fs = float(fs)
        # Basically the absolute path of the file (minus the drive name) adds that to your current directory. Makes all dirs needed
        pOut = os.path.join(picture_out, file[:-4] + "_" + file2[:-4] + '.png')
        sOut = os.path.join(sound_out, file[:-4] + "_" + file2[:-4] + '.wav')
        sonify(TheData[0], fs, sOut, pOut, N=5)


