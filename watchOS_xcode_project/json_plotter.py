import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load(path):
    with open(path) as f:
        data = json.load(f)
    return data

mel_gram = load('./mel_gram.json')

STFT = np.asarray(mel_gram).T
plt.imshow(STFT, cmap='hot', interpolation='nearest')
plt.show()
