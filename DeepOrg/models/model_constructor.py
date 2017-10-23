from keras.layers import concatenate


def merge_multiple_bases(base_models, input_names, trainable, logging):
    base_model_inputs = []
    base_model_outputs = []

    assert(len(base_models) == len(input_names))

    for i in range(len(base_models)):
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_models[i].layers:
            layer.name = input_names[i] + '_' + layer.name
            layer.trainable = trainable

        base_model_inputs.append(base_models[i].input)
        base_model_outputs.append(base_models[i].output)

    multiple_bases = concatenate(base_model_outputs, axis=1)
    logging.info(str(base_model_outputs[0]._keras_shape) + ' ' + str(multiple_bases._keras_shape))

    return base_model_inputs, multiple_bases
