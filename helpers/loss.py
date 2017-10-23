import numpy as np
import keras.backend as K
from itertools import product


def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    return K.categorical_crossentropy(y_pred, y_true) * final_mask


# def w_categorical_crossentropy(y_true, y_pred, weights):
#     nb_cl = len(weights)
#     final_mask = K.zeros_like(y_pred[:, 0])
#     y_pred_max = K.max(y_pred, axis=-1)
#     y_pred_max = K.expand_dims(y_pred_max, axis=-1)
#     y_pred_max_mat = K.equal(y_pred, y_pred_max)
#
#     for c_p, c_t in product(range(nb_cl), range(nb_cl)):
#         final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
#     return K.categorical_crossentropy(y_pred, y_true) * final_mask


class WeightedCategoricalCrossEntropy(object):

  def __init__(self, weights):
    nb_cl = len(weights)
    self.weights = np.ones((nb_cl, nb_cl))
    for class_idx, class_weight in weights.items():
        self.weights[0][class_idx] = class_weight
        self.weights[class_idx][0] = class_weight
    self.__name__ = 'w_categorical_crossentropy'

  def __call__(self, y_true, y_pred):
    return self.w_categorical_crossentropy(y_true, y_pred)

  def w_categorical_crossentropy(self, y_true, y_pred):
    nb_cl = len(self.weights)
    final_mask = K.zeros_like(y_pred[..., 0])
    y_pred_max = K.max(y_pred, axis=-1)
    y_pred_max = K.expand_dims(y_pred_max, axis=-1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        w = K.cast(self.weights[c_t, c_p], K.floatx())
        y_p = K.cast(y_pred_max_mat[..., c_p], K.floatx())
        y_t = K.cast(y_pred_max_mat[..., c_t], K.floatx())
        final_mask += w * y_p * y_t
    return K.categorical_crossentropy(y_pred, y_true) * final_mask
