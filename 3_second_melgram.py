import librosa
import numpy as np
import os
import glob
import sys
import matplotlib.pyplot as plt

def spectrogram_image(y, sr, out, hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                            n_fft=hop_length*4, hop_length=hop_length)

    mels = librosa.power_to_db(mels, ref=np.max)
    mels = np.flipud(mels)

    plt.imshow(mels, cmap='hot', interpolation='nearest')
    plt.axis('off')

    if(mels.shape == (128, 94)):
        np.save(out, mels) # save NPY mel spectrogram array

        # Uncomment next line to also save PNG image to activity folders
        #plt.savefig('./nparrays/Activity ' + out[17:18] + '/' + out[9:] + '.png', format='png', bbox_inches='tight')

    plt.show()
    plt.clf()

if __name__ == '__main__':
     for fp in os.listdir('./audio'):
        if fp.endswith(".wav"):
            # settings
            fft_length = 2048 # length of each fft window
            hop_length = 512 # number of samples per time-step in spectrogram
            n_mels = 128 # number of bins in spectrogram. Height of image
            time_steps = 2149 # number of time-steps. Width of image

            # load audio
            path = './audio/' + fp
            y, sr = librosa.load(path, offset=0.0, duration=3.0, sr=16000)
            y = librosa.util.normalize(y)
            out = 'nparrays_4class/' + fp[:-4]

            # convert to NPY
            spectrogram_image(y, sr=sr, out=out, hop_length=hop_length, n_mels=n_mels)
            print('wrote file', out)
