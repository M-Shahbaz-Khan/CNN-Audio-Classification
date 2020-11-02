import librosa
import numpy as np
import matplotlib.pyplot as plt
import json, codecs

mel_basis = librosa.filters.mel(16000, 1023)
file_path = "./mel_basis_02092020.json"
mel_basis2 = mel_basis.tolist()

with open(file_path, 'w', encoding='utf-8') as f:
	    json.dump(mel_basis2, f, ensure_ascii=False, indent=4)

print(mel_basis.shape)

#json.dump(mel_basis, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=False, indent=4)
plt.imshow(mel_basis2, cmap='hot', interpolation='nearest')
plt.show()
