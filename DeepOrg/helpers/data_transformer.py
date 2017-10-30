import random
import numpy as np

from keras.utils import np_utils


def transform(X_train, y_train, X_val, y_val, nb_classes):
    if type(X_train) is list:
        c = list(zip(X_train, y_train))
        random.shuffle(c)
        X_train, y_train = zip(*c)

        c = list(zip(X_val, y_val))
        random.shuffle(c)
        X_val, y_val = zip(*c)

    elif type(X_train) is np.ndarray:
        indices = np.random.permutation(X_train.shape[0])
        X_train = X_train[indices]
        y_train = y_train[indices]

        indices = np.random.permutation(X_val.shape[0])
        X_val = X_val[indices]
        y_val = y_val[indices]

    # random.shuffle(X_train_source)
    # random.shuffle(X_val_source)

    # Convert class vectors to binary class matrices.
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_val = np_utils.to_categorical(y_val, nb_classes)

    return X_train, Y_train, X_val, Y_val
