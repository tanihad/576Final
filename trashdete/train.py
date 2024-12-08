from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np


def train_test(data, labels, epochs, batch_size):

    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    # turn labels to one-hot encoding
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    (X_train, X_test, y_train, y_test) = train_test_split(data, labels,
        test_size=0.15, stratify=labels, random_state=158)

    aug = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)

    # can add early stopping and other metrics here
    h = model.fit(
        generator=aug.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=(X_test, y_test),
        validation_steps=len(X_test) // batch_size,
        epochs=epochs)

    logits = model.predict(X_test, batch_size=batch_size)
    pred= np.argmax(logits, axis=1)

    return h, logits, pred