# Convert .h5 Model to CoreML
#
# Note that it will only work if only supported keras layers are used
# in the model, and only if supported versions of keras and tensorflow were
# used in compiling the model. Tested with Keras 2.2.4 and TF 1.13.2
#
# coremltools compatibility: https://coremltools.readme.io/docs

from keras.models import load_model
import coremltools

model = load_model('./CNN_evaluation/model.h5')

output_labels = ['Class Management', 'Lecture', 'Practice', 'Q&A']
your_model = coremltools.converters.keras.convert(model, input_names=['STFT'],
                                                  output_names=['Classroom Activities'],
                                                  class_labels=output_labels)
your_model.author = 'Imran Zualkernan and Muhammed Khan'
your_model.short_description = 'Classifier - Mel Spectrograms of Classroom Audio'
your_model.input_description['STFT'] = 'Takes as input a mel-scaled STFT'

your_model.save('./CNN_evaluation/classroomClassifier_06242020.mlmodel')
