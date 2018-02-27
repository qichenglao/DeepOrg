from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, Input, merge, Lambda, BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D


def load_alexnet(nb_classes, img_rows, img_cols, img_channels):
    inputs = Input(shape=(img_channels, img_rows, img_cols))

    # Layer 1
    conv_1 = Convolution2D(96, (11, 11), activation='relu', padding='same')(inputs)
    conv_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(conv_1)
    conv_1 = BatchNormalization()(conv_1)

    # Layer 2
    conv_2 = Convolution2D(256, (5, 5), activation='relu', padding='same')(conv_1)
    conv_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(conv_2)
    conv_2 = BatchNormalization()(conv_2)

    # Layer 3
    conv_3 = Convolution2D(384, (3, 3), activation='relu', padding='same')(conv_2)

    # Layer 4
    conv_4 = Convolution2D(384, (3, 3), activation='relu', padding='same')(conv_3)

    # Layer 5
    conv_5 = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv_4)
    conv_5 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(conv_5)

    # Layer 6
    dense_1 = Flatten()(conv_5)
    dense_1 = Dense(4096, activation='relu', kernel_initializer='he_normal')(dense_1)

    # Layer 7
    dense_2 = Dense(4096, activation='relu', kernel_initializer='he_normal')(dense_1)

    # Layer 8
    dense_3 = Dense(nb_classes, activation='relu', kernel_initializer='he_normal')(dense_2)
    prediction = Activation("softmax")(dense_3)

    alexnet = Model(inputs=inputs, outputs=prediction)

    return alexnet


#############################################################################################################
# def mean_subtract(img):
#     img = T.set_subtensor(img[:, 0, :, :], img[:, 0, :, :] - 123.68)
#     img = T.set_subtensor(img[:, 1, :, :], img[:, 1, :, :] - 116.779)
#     img = T.set_subtensor(img[:, 2, :, :], img[:, 2, :, :] - 103.939)
#
#     return img / 255.0


# def crosschannelnormalization(alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
#     """
#     This is the function used for cross channel normalization in the original
#     Alexnet
#     """
# 
#     def f(X):
#         b, ch, r, c = X.shape
#         half = n // 2
#         square = K.square(X)
#         extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0, 2, 3, 1))
#                                               , (0, half))
#         extra_channels = K.permute_dimensions(extra_channels, (0, 3, 1, 2))
#         scale = k
#         for i in range(n):
#             scale += alpha * extra_channels[:, i:i + ch, :, :]
#         scale = scale ** beta
#         return X / scale
# 
#     return Lambda(f, output_shape=lambda input_shape: input_shape, **kwargs)


# def splittensor(axis=1, ratio_split=1, id_split=0, **kwargs):
#     def f(X):
#         div = X.shape[axis] // ratio_split
# 
#         if axis == 0:
#             output = X[id_split * div:(id_split + 1) * div, :, :, :]
#         elif axis == 1:
#             output = X[:, id_split * div:(id_split + 1) * div, :, :]
#         elif axis == 2:
#             output = X[:, :, id_split * div:(id_split + 1) * div, :]
#         elif axis == 3:
#             output = X[:, :, :, id_split * div:(id_split + 1) * div]
#         else:
#             raise ValueError('This axis is not possible')
# 
#         return output
# 
#     def g(input_shape):
#         output_shape = list(input_shape)
#         output_shape[axis] = output_shape[axis] // ratio_split
#         return tuple(output_shape)
# 
#     return Lambda(f, output_shape=lambda input_shape: g(input_shape), **kwargs)


# def get_alexnet(input_shape, nb_classes, mean_flag):
#     # code adapted from https://github.com/heuritech/convnets-keras
# 
#     inputs = Input(shape=input_shape)
# 
#     if mean_flag:
#         mean_subtraction = Lambda(mean_subtract, name='mean_subtraction')(inputs)
#         conv_1 = Convolution2D(96, 11, 11, subsample=(4, 4), activation='relu',
#                                name='conv_1', init='he_normal')(mean_subtraction)
#     else:
#         conv_1 = Convolution2D(96, 11, 11, subsample=(4, 4), activation='relu',
#                                name='conv_1', init='he_normal')(inputs)
# 
#     conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
#     conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
#     conv_2 = ZeroPadding2D((2, 2))(conv_2)
#     conv_2 = merge([
#         Convolution2D(128, 5, 5, activation="relu", init='he_normal', name='conv_2_' + str(i + 1))(
#             splittensor(ratio_split=2, id_split=i)(conv_2)
#         ) for i in range(2)], mode='concat', concat_axis=1, name="conv_2")
# 
#     conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
#     conv_3 = crosschannelnormalization()(conv_3)
#     conv_3 = ZeroPadding2D((1, 1))(conv_3)
#     conv_3 = Convolution2D(384, 3, 3, activation='relu', name='conv_3', init='he_normal')(conv_3)
# 
#     conv_4 = ZeroPadding2D((1, 1))(conv_3)
#     conv_4 = merge([
#         Convolution2D(192, 3, 3, activation="relu", init='he_normal', name='conv_4_' + str(i + 1))(
#             splittensor(ratio_split=2, id_split=i)(conv_4)
#         ) for i in range(2)], mode='concat', concat_axis=1, name="conv_4")
# 
#     conv_5 = ZeroPadding2D((1, 1))(conv_4)
#     conv_5 = merge([
#         Convolution2D(128, 3, 3, activation="relu", init='he_normal', name='conv_5_' + str(i + 1))(
#             splittensor(ratio_split=2, id_split=i)(conv_5)
#         ) for i in range(2)], mode='concat', concat_axis=1, name="conv_5")
# 
#     dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5)
# 
#     dense_1 = Flatten(name="flatten")(dense_1)
#     dense_1 = Dense(4096, activation='relu', name='dense_1', init='he_normal')(dense_1)
#     dense_2 = Dropout(0.5)(dense_1)
#     dense_2 = Dense(4096, activation='relu', name='dense_2', init='he_normal')(dense_2)
#     dense_3 = Dropout(0.5)(dense_2)
#     dense_3 = Dense(nb_classes, name='dense_3_new', init='he_normal')(dense_3)
# 
#     prediction = Activation("softmax", name="softmax")(dense_3)
# 
#     alexnet = Model(input=inputs, output=prediction)
# 
#     return alexnet

#############################################################################################################
