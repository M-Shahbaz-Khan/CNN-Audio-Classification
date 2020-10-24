# Set epochs to 40, fix test generator indices

import keras
import tensorflow as tf
from keras import regularizers, optimizers
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, InputLayer, GaussianNoise
from keras.layers import Conv2D, MaxPooling2D, Convolution2D
from keras.optimizers import SGD, Adam, Adamax
from keras.callbacks import EarlyStopping
from keras.callbacks import History
from keras.utils import plot_model, to_categorical
from keras.regularizers import l1
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import os, os.path
import csv
import shutil
import sys
from PIL import Image

eval_path = './CNN_evaluation/'
melspec_path = './nparrays_4class'

if not os.path.isdir(eval_path):
    os.makedirs(eval_path)

if os.path.exists(eval_path + '/evaluation_10_fold.txt'):
    os.remove(eval_path + '/evaluation_10_fold.txt')

def pre_process_oversample(foldNum):
    # Fold code unnecessary, remove

    x_train_list = []
    y_train_list = []
    xy_validate_list = []
    x_validate_list = []
    y_validate_list = []
    x_test_list = []
    y_test_list = []

    smote_y_label = []

    ################################################
    foldsProcessed = 0
    ################################################

    while(foldsProcessed < 10): # Build train data
        if foldNum == 11:
            foldNum = 1

        currentDirectory = melspec_path + '/fold' + str(foldNum)

        for filename in os.listdir(currentDirectory):
            if filename.endswith('.npy'):
                new_sample = np.load(currentDirectory + '/' + filename)

                x_train_list.append(new_sample)

                currentActivity = int(filename[8:9])
                smote_y_label.append(currentActivity)

                if (currentActivity == 2):
                    y_train_list.append(np.array([1,0,0,0]))
                else:
                    if (currentActivity == 3):
                        y_train_list.append(np.array([0,1,0,0]))
                    else:
                        if (currentActivity == 4):
                            y_train_list.append(np.array([0,0,1,0]))
                        else:
                            if (currentActivity == 5):
                                y_train_list.append(np.array([0,0,0,1]))

        foldNum += 1
        foldsProcessed += 1

    x_train = np.expand_dims(np.asarray(x_train_list), axis=3)
    y_train = np.asarray(y_train_list)

    print('Before oversampling: ' + str(x_train.shape) + ' - ' + str(y_train.shape))

    # Split valid and train
    X, x_validate, y, y_validate = train_test_split(x_train, y_train, test_size=0.2, random_state=527)
    xy_validate = (x_validate, y_validate)

    # split test and train
    X, x_test, y, y_test = train_test_split(X, y, test_size=0.25, random_state=527)

    # oversample train
    ros = RandomOverSampler(random_state=42)
    nsamples, ny, nx, nfloat = X.shape
    X = X.reshape((nsamples, nx*ny*nfloat))
    x_ros, y_ros = ros.fit_resample(X, y)
    ros_samples, _, = x_ros.shape
    x_ros = x_ros.reshape(ros_samples, ny, nx, nfloat)

    return x_ros, y_ros, xy_validate, x_test, y_test

def train_and_evaluate(foldNum, x_train, y_train, xy_validate, x_test, y_test):
    model = compile_model()
    history, model = train(model, x_train, y_train, xy_validate, x_test, y_test)
    plot_ROC(model, foldNum, x_test, y_test)
    #if(foldNum == 1):
        #plot_model(model, to_file= eval_path + '/_model.png')
    plot_learning_curve(history, foldNum)
    save(model, 'model')

def train(model, x_train, y_train, xy_validate, x_test, y_test):
    history = History()
    earlystop = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min', restore_best_weights=True)
    model.fit(x=x_train, y=y_train, batch_size=10, epochs=1, verbose=1, callbacks=[earlystop, history], validation_data=xy_validate)
    return history, model

# To see images in batch 1 of train set
def see_train_images():
    x, y = train_generator.next()
    for i in range(0,8):
        image = x[i]
        plt.title(y[i])
        plt.imshow(image)
        plt.show()

# To see images in test set
def see_train_images():
    test_generator.reset()
    x, y = test_generator.next()
    for i in range(0,8):
        image = x[i]
        plt.title(y[i])
        plt.imshow(image)
        plt.show()

def plot_learning_curve(history, foldNum):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'], lw=2)
    plt.plot(history.history['val_acc'], lw=2)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig(eval_path + '/acc_curve.png')
    plt.clf()

    # Plot training & validation loss values
    plt.plot(history.history['loss'], lw=2)
    plt.plot(history.history['val_loss'], lw=2)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig(eval_path + '/loss_curve.png')
    plt.clf()

def compile_model():
    inputShape = (128, 94, 1)
    model = Sequential()
    # Removed InputLayer to allow conversion to CoreML (might be fixable by downgrading keras and tensorflow)
    # model.add(InputLayer(input_shape=inputShape))

    model.add(Conv2D(128, 9, strides=1, activation='relu',padding='same',input_shape=inputShape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(96,  5, strides=1, activation='relu',padding = 'same'))
    model.add(BatchNormalization())

    model.add(Conv2D(96,  5, strides=1, activation='relu',padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.16))

    model.add(Conv2D(32,  11, strides=1, activation='relu',padding = 'same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.37))

    model.add(Conv2D(64,  5, strides=1, activation='relu',padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(keras.layers.Dense(units = 64, activation = 'relu', activity_regularizer=l1(2.45e-05)))
    model.add(Dense(4, activation='softmax'))

    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    model.compile(
                optimizer=keras.optimizers.Adam(
                1.66e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    model.summary()

    return model

def save(model, modelName):
    modelName = eval_path + '/' + modelName + '.h5'
    model.save(modelName)

def load(modelName):
    from keras.models import load_model
    model = load_model(modelName)
    return model

def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel_initializer.run(session=session)
        if hasattr(layer, 'bias_initializer'):
            layer.bias_initializer.run(session=session)

def plot_ROC(model, foldNum, x_test, y_test):
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from itertools import cycle

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 4

    y_score=model.predict(x_test, batch_size=None, verbose=0, steps=None)

    prediction_classes=[]

    rowNumber = 1
    # replace with code to get highest value in array iterating over y_score, much faster
    for row in y_score:
        print('Processing ' + str(rowNumber) + ': ' + str(row))
        rowNumber += 1
        pred_class=np.argmax(row)
        prediction_classes.append(pred_class + 2)

    y_test_classnames = []

    for row in y_test:
        y_test_classnames.append(np.argmax(row) + 2)

    with open(eval_path + '/evaluation_10_fold.txt', "a") as myfile:
        target_names = ['Class Mgmt. (2)', 'Lecture (3)', 'Practice (4)', 'Q&A (5)']

        myfile.write("Fold " + str(foldNum) + "\n\n")
        myfile.write(classification_report(y_test_classnames, prediction_classes, target_names=target_names))
        myfile.write("\n\n")

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['darkorange', 'cornflowerblue', 'darkgreen', 'pink'])

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i + 2, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(eval_path + '/ROC.png')
    plt.clf()

    cm = confusion_matrix(y_test_classnames, prediction_classes)

    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j,i,cm[i, j],
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        axes = plt.gca()
        axes.set_ylim([-0.5,3.5])

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(eval_path + '/CM.png')
        plt.clf()

    CMlabels = ['2', '3', '4', '5']
    plot_confusion_matrix(cm, CMlabels, normalize=False)


for i in range(1, 11):
    x_train, y_train, xy_validate, x_test, y_test = pre_process_oversample(i)
    train_and_evaluate(i, x_train, y_train, xy_validate, x_test, y_test)
    break
