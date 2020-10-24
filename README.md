# CNN Audio Classification

Audio Classification using CNNs with mel spectrograms and inference on Apple Watch with CoreML.

To run melgram_generator.py you will need the audio data, which can be downloaded here: https://drive.google.com/file/d/1-Sr1tTDXBA6-kyN0XTrujAFxDBEqJqZT/view?usp=sharing. Place audio folder in project root.

To run CNN_train.py or CNN_hyperparemeter_search.py you will need the .npy mel spectrogram data, which you can either generate yourself from the audio using melgram_genrator.py or download here: https://drive.google.com/file/d/1XIkZ36yo7ZE6i6KqJDCUBHU2uL92DGRD/view?usp=sharing. Place the downloaded folder nparrays_4class in the same directory as the python files.

Running CNN_hyperparemeter_search.py will give you a list of the 10 best sets of hyperparameters found, and checkpoints will be saved in a folder called grid_search. This can be used if you want to halt the hyperparameter search without letting it run to completion.

Running CNN_train.py will train and evaluate a CNN on the .npy data. The results and the model will be saved in the CNN_evaluation folder. converter.py can then be run to convert the .h5 model to the .mlmodel format.

Tested with:
  - Keras 2.2.4
  - Tensorflow-gpu 1.13.2 (CUDA 10.0/Cudnn 7.4.2.24)
  - Latest versions of all other packages as of 10/24/2020
