import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
from matplotlib import pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt

model_state = torch.load("./models/emotion_detection_model_state.pth", map_location=torch.device('cpu'))
class_labels = ["Angry","Disgust","Fear", "Happy", "Neutral", "Sad", "Suprise"]

def load_dataset(net=True):
    """Utility function to load the FER2013 dataset.
    
    It returns the formated tuples (X_train, y_train) , (X_test, y_test).

    Parameters
    ==========
    net : boolean
        This parameter is used to reshape the data from images in 
        (cols, rows, channels) format. In case that it is False, a standard
        format (cols, rows) is used.
    """

    # Load and filter in Training/not Training data:
    df = pd.read_csv('fer2013.csv')
    training = df.loc[df['Usage'] == 'Training']
    testing = df.loc[df['Usage'] != 'Training']

    # X_train values:
    X_train = training[['pixels']].values
    X_train = [np.fromstring(e[0], dtype=int, sep=' ') for e in X_train]
    if net:
        X_train = [e.reshape((48, 48, 1)).astype('float32') for e in X_train]
    else:
        X_train = [e.reshape((48, 48)) for e in X_train]
    X_train = np.array(X_train)

    # X_test values:
    X_test = testing[['pixels']].values
    X_test = [np.fromstring(e[0], dtype=int, sep=' ') for e in X_test]
    if net:
        X_test = [e.reshape((48, 48, 1)).astype('float32') for e in X_test]
    else:
        X_test = [e.reshape((48, 48)) for e in X_test]
    X_test = np.array(X_test)

    # y_train values:
    y_train = training[['emotion']].values
    y_train = keras.utils.to_categorical(y_train)

    # y_test values
    y_test = testing[['emotion']].values
    y_test = keras.utils.to_categorical(y_test)

    return (X_train, y_train) , (X_test, y_test)

(X_train, y_train) , (X_test, y_test) = load_dataset()

model = ResNet()
model.load_state_dict(model_state)

# Predict using trained model:
y_pred = model.predict(X_test)
y_pred = np.asarray([np.argmax(e) for e in y_pred])
y_true = np.asarray([np.argmax(e) for e in y_test])

# Compute confusion matrix:
cm = confusion_matrix(y_true, y_pred)
cm_normalised = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot confusion matrix:
sns.set(font_scale=1.5) 
fig, ax = plt.subplots(figsize=(10,10))
ax = sns.heatmap(cm_normalised, annot=True, linewidths=0, square=False, 
                 cmap='gray', yticklabels=class_labels,
                 xticklabels=class_labels, vmin=0,
                 vmax=np.max(cm_normalised), fmt=".2f",
                 annot_kws={"size": 20})
ax.set(xlabel='Predicted label', ylabel='True label')