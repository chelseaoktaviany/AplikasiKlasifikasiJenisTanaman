#untuk direktori
import os, glob, json, sys
import cv2
import numpy as np

#library untuk membuat visualisasi hasil akurasi
import matplotlib.pyplot as plt

#optimizing
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.pop('TF_CONFIG', None)

if '.' not in sys.path:
  sys.path.insert(0, '.')


#library untuk menerapkan algoritma cnn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Activation, Flatten, Dense

#PROSES PELATIHAN MODEL

batch_size = 8

#CNN untuk 4 kelas
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(200,200,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('softmax'))

model.summary()

from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_set = train_datagen.flow_from_directory('dataset/train',
                                     target_size=(200,200),
                                     batch_size=batch_size,
                                     class_mode='categorical',
                                     shuffle=True
                                     )

val_set = val_datagen.flow_from_directory('dataset/validation',
                                     target_size=(200,200),
                                     batch_size=batch_size,
                                     class_mode='categorical',
                                     shuffle = True
                                    )

#proses training
history = model.fit(train_set,
                    steps_per_epoch=15,
                    epochs = 50,
                    #validation_data = val_set,
                    #validation_steps= 4,
                    verbose = 1
                    )

#menyimpan model
model.save('leafClassification.h5')

#viewing history
# Loss Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['loss'], 'r', linewidth=3.0)
plt.legend(['Training loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)
plt.show()

# Accuracy Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['accuracy'], 'r', linewidth=3.0)
plt.legend(['Training Accuracy'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)
plt.show()

#PROSES PENGUJIAN MODEL
test_datagen = ImageDataGenerator(rescale=1. / 255,
                                 shear_range = 0.2,
                                 zoom_range = 0.2,
                                 horizontal_flip = True
                                 )
test_generator = test_datagen.flow_from_directory('dataset/test/',
                                                  target_size=(200,200),
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=True)

#mendapatkan nilai akurasi untuk data testing
#test_score = model.evaluate(test_generator)

#print("[INFO] accuracy: {:.2f}%".format(test_score[1] * 100))
#print("[INFO] Loss: ",test_score[0])

#membuat matrix confusion
# Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Print the Target names
target_names = []
for key in train_set.class_indices:
    target_names.append(key)
print(target_names)

Y_pred = model.predict(test_generator,
                                 steps=np.ceil(test_generator.samples / test_generator.batch_size),
                                 verbose=1, workers=0)
y_pred = np.argmax(Y_pred, axis=1)

print("True label: ", test_generator.classes)
print("Predicted label: ", y_pred)

#confusion matrix
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
        print("Normalized confusion matrix")

    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.

    import itertools
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

cm = confusion_matrix(test_generator.classes, y_pred)
plot_confusion_matrix(cm, target_names, title='Confusion Matrix')

# Print Classification Report
print('Classification Report')
print(classification_report(test_generator.classes, y_pred, target_names=target_names))