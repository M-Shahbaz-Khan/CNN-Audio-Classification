# CNN Audio Classification - watchOS inference

xcode_project contains a Swift 5 implementation of an application that records audio and makes class/activity predictions using the trained CNN model converted to CoreML .mlmodel format.

mel_basis_generator.py is used to generate a mel basis in order to save computation time when running on the watch. Its purpose is to scale the STFT created from raw audio to convert it to a mel spectrogram. This could otherwise be implemented as a one-time calculation within the WatchOS app that is run on start up.

Tested on a Series 4 Apple Watch running WatchOS 5.
