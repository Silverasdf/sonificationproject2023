# Add spectrograms - Takes two random files from directory of .mat files, adds them together, and makes the spectrograms
# Ryan Peruski, 07/24/23
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
from sonifyfn import sonify

data_dir = '/root/SonificationProject/Data/SomeMatFiles'
picture_out = '/root/SonificationProject/Data/sonified_spectrograms_on'
sound_out = '/root/SonificationProject/Data/sonified_spectrograms_on'
num_files = 100

#Choose random file from directory
for i in range(num_files):
    eligible_files = [file for file in os.listdir(data_dir) if file.endswith('1.mat')]
    file = np.random.choice(eligible_files)
    filepath = os.path.join(data_dir, file).replace('\\', '/')

    file2 = np.random.choice(eligible_files)
    filepath2 = os.path.join(data_dir, file).replace('\\', '/')

    sample = sio.loadmat(filepath)
    sample2 = sio.loadmat(filepath2)
    TheData = 0.5 * (sample['data'].astype(float) + sample2['data'].astype(float))
    fs = sample['samp_rate'].item()
    fs = float(fs)
    # Basically the absolute path of the file (minus the drive name) adds that to your current directory. Makes all dirs needed
    pOut = os.path.join(picture_out, file[:-4] + "_" + file2[:-4] + '.png')
    sOut = os.path.join(sound_out, file[:-4] + "_" + file2[:-4] + '.wav')
    sonify(TheData, fs, sOut, pOut, N=5)


