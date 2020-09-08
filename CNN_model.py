# This script was written in google colab jupyter notebook environment,
# some functions might be environment-specific.

# imports

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import io
import keras
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.python.keras import Sequential
from google.colab import files
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, SCORERS, auc, confusion_matrix, f1_score, classification_report, multilabel_confusion_matrix

# display options

np.set_printoptions(suppress=True)

# read numpy arrays

uploaded = files.upload()
X = np.load(io.BytesIO(uploaded['80000_32_32.npy']))

uploaded = files.upload()
y = np.load(io.BytesIO(uploaded['80000_1.npy']))

# Illustration of a session in a gray-scale picture format

%matplotlib inline
plt.imshow(X[567], cmap="gray")

# required formatting 

y = keras.utils.to_categorical(y, 2)
X = X.reshape(80000,32,32,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# building the model with keras sequential function
# hyper-parameters are obtained beforehand using hyperas hyper-parameter optimization library for keras

model = Sequential()
model.add(Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=(32,32,1)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5442420919810019))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.43404126142165134))

model.add(Flatten())
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.6975075528853121))
model.add(Dense(2, activation='softmax'))

model.summary()

# Model compilation and training

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])
hist = model.fit(X_train, y_train, epochs=7, batch_size=256, shuffle=True, validation_data=(X_test, y_test))

# Illustration of loss per epoch

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# Illustration of accuracy per epoch

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

# conversion of categorical y value to scalar value

y_pred_keras = model.predict(X_test)
y_pred_keras_f1 = (y_pred_keras > 0.5)
y_pred_keras_f1 = np.argmax(y_pred_keras_f1, axis=1)
y_test_f1 = np.argmax(y_test, axis=1)

# printing the performance metric results

print("=== Confusion Matrix ===")
np = confusion_matrix(y_test_f1, y_pred_keras_f1)
df_cm = pd.DataFrame(np, index=["0", "1"], columns=["0","1"])
print(df_cm)
print('\n')
print("=== F1 Score ===")
print(f1_score(y_test_f1, y_pred_keras_f1))
print('\n')
roc_auc = roc_auc_score(y_test, y_pred_keras)
print("=== RoC Result for CNN Classifier ===")
print(roc_auc)

# Illustration of ROC AUC score

probs = model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(np.argmax(y_test, axis=1), preds)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Note that * symbols were placed where required to mask personal and institutional information.
