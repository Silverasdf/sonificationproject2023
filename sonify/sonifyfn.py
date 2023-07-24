import numpy as np
import scipy.io.wavfile as wavfile
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
import os

def create_directories(filepath):
    # Get the current directory
    current_directory = os.getcwd()

    # Create the destination directory relative to the current directory
    relative_path = os.path.relpath(filepath, start=current_directory)
    destination_directory = os.path.dirname(relative_path)

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_directory, exist_ok=True)

    # Create the new file path with the '.wav' extension
    new_filepath = os.path.join(destination_directory, os.path.splitext(os.path.basename(relative_path))[0] + ".wav")

    return new_filepath

def sonify(TheData, fs, sOut, pOut, N):
    dataLow = TheData
    # pretend the SR is 2e6/10
    fs = 2e6 / 40

    dataLow = signal.lfilter([100], 1, dataLow)

    window = signal.hann(int(fs / 4))  # Generate 1-D Hanning window

    # data, fs, window, nfft, noverlap
    Length = int(N * fs)  # Change to  seconds
    dataLowPart = dataLow[:Length]

    f, t, Sxx = signal.spectrogram(dataLowPart, fs, window, nfft=int(fs / 4), noverlap=int(fs / 8))

    plt.figure()
    plt.imshow(np.log(np.abs(Sxx[:1000, :])).T, aspect='auto', cmap='jet', extent=[t.min(), t.max(), f.min(), f.max()])
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Spectrogram')
    plt.xlim([t.min(), N])
    plt.colorbar()
    plt.savefig(pOut)
    plt.close()

    # Adjust the number of samples for 5 seconds duration
    num_samples = int(fs * N)
    dataLowPartNorm = dataLowPart[:num_samples] / np.max([np.max(dataLowPart), np.abs(np.min(dataLowPart))])

    # Convert dataLowPartNorm to int16 before writing to WAV file
    dataLowPartNorm = (dataLowPartNorm * np.iinfo(np.int16).max).astype(np.int16)

    # Reshape the dataLowPartNorm array to (n_samples, 1)
    dataLowPartNorm = np.reshape(dataLowPartNorm, (-1, 1))
    try:
        wavfile.write(sOut, int(fs), dataLowPartNorm[:num_samples])
    except Exception as e:
        print(f'Error: {e}. Returning...')
        return