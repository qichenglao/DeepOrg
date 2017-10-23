from keras import applications
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, concatenate
from keras.models import Model

from models import model_logger

img_channels = 3


def get_ResNet50(model_name, nb_classes, img_rows, img_cols, logging, logging_msg):
    base_model_40 = applications.ResNet50(include_top=False, weights='imagenet',
                                          input_shape=(img_channels, img_rows, img_cols))
    base_model_100 = applications.ResNet50(include_top=False, weights='imagenet',
                                           input_shape=(img_channels, img_rows, img_cols))
    base_model_200 = applications.ResNet50(include_top=False, weights='imagenet',
                                           input_shape=(img_channels, img_rows, img_cols))
    base_model_400 = applications.ResNet50(include_top=False, weights='imagenet',
                                           input_shape=(img_channels, img_rows, img_cols))

    x_40 = base_model_40.output
    x_100 = base_model_100.output
    x_200 = base_model_200.output
    x_400 = base_model_400.output

    x = concatenate([x_40, x_100, x_200, x_400], axis=1)
    logging.info(str(x_40._keras_shape) + ' ' + str(x._keras_shape))

    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)
    # print(x._keras_shape)
    # x = Flatten()(x)

    x = Dropout(0.5)(x)
    # let's add a fully-connected layer
    x = Dense(256, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(nb_classes, activation='softmax')(x)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model_40.layers:
        layer.name = '40_' + layer.name
        layer.trainable = False
    for layer in base_model_100.layers:
        layer.name = '100_' + layer.name
        layer.trainable = False
    for layer in base_model_200.layers:
        layer.name = '200_' + layer.name
        layer.trainable = False
    for layer in base_model_400.layers:
        layer.name = '400_' + layer.name
        layer.trainable = False

    # this is the model we will train
    model = Model(inputs=[base_model_40.input, base_model_100.input, base_model_200.input, base_model_400.input],
                  outputs=predictions)

    model_logger.summary(model, model_name, logging_msg)
    return model


def get_VGG19(model_name, nb_classes, img_rows, img_cols, logging, logging_msg):
    base_model_40 = applications.VGG19(include_top=False, weights='imagenet',
                                       input_shape=(img_channels, img_rows, img_cols))
    base_model_100 = applications.VGG19(include_top=False, weights='imagenet',
                                        input_shape=(img_channels, img_rows, img_cols))
    base_model_200 = applications.VGG19(include_top=False, weights='imagenet',
                                        input_shape=(img_channels, img_rows, img_cols))
    base_model_400 = applications.VGG19(include_top=False, weights='imagenet',
                                        input_shape=(img_channels, img_rows, img_cols))

    x_40 = base_model_40.output
    x_100 = base_model_100.output
    x_200 = base_model_200.output
    x_400 = base_model_400.output

    x = concatenate([x_40, x_100, x_200, x_400], axis=1)
    logging.info(str(x_40._keras_shape) + ' ' + str(x._keras_shape))

    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(256, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(nb_classes, activation='softmax')(x)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model_40.layers:
        layer.name = '40_' + layer.name
        layer.trainable = False
    for layer in base_model_100.layers:
        layer.name = '100_' + layer.name
        layer.trainable = False
    for layer in base_model_200.layers:
        layer.name = '200_' + layer.name
        layer.trainable = False
    for layer in base_model_400.layers:
        layer.name = '400_' + layer.name
        layer.trainable = False

    # this is the model we will train
    model = Model(inputs=[base_model_40.input, base_model_100.input, base_model_200.input, base_model_400.input],
                  outputs=predictions)

    model_logger.summary(model, model_name, logging_msg)
    return model


# build the network
def get_inception_v3(model_name, nb_classes, img_rows, img_cols, logging, logging_msg):
    base_model_40 = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet',
                                                          input_shape=(img_channels, img_rows, img_cols))
    base_model_100 = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet',
                                                           input_shape=(img_channels, img_rows, img_cols))
    base_model_200 = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet',
                                                           input_shape=(img_channels, img_rows, img_cols))
    base_model_400 = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet',
                                                           input_shape=(img_channels, img_rows, img_cols))

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model_40.layers:
        layer.name = '40_' + layer.name
        layer.trainable = False
    for layer in base_model_100.layers:
        layer.name = '100_' + layer.name
        layer.trainable = False
    for layer in base_model_200.layers:
        layer.name = '200_' + layer.name
        layer.trainable = False
    for layer in base_model_400.layers:
        layer.name = '400_' + layer.name
        layer.trainable = False

    x_40 = base_model_40.output
    x_100 = base_model_100.output
    x_200 = base_model_200.output
    x_400 = base_model_400.output

    x = concatenate([x_40, x_100, x_200, x_400], axis=1)
    logging.info(str(x_40._keras_shape) + ' ' + str(x._keras_shape))

    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)

    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(nb_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=[base_model_40.input, base_model_100.input, base_model_200.input, base_model_400.input], outputs=predictions)

    model_logger.summary(model, model_name, logging_msg)
    return model
