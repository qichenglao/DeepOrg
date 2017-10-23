import numpy as np

from keras import backend as K


def summary(model, model_name, logging_msg):
    logging_msg.info('################################')
    logging_msg.info('Model %s Summary:' % model_name)
    logging_msg.info('################################')
    model.summary(print_fn=lambda x: logging_msg.info(x))
    # print(model.summary(), file=output)


def layer_list(model, model_name, logging_msg):
    logging_msg.info('################################')
    logging_msg.info('Model %s Layers:' % model_name)
    logging_msg.info('################################')
    for i, layer in enumerate(model.layers):
        logging_msg.info(str(i) + ' ' + layer.name)


def parameter_numbers(model, model_name, logging_msg):
    trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

    logging_msg.info('################################')
    logging_msg.info('Model %s Parameters:' % model_name)
    logging_msg.info('################################')
    logging_msg.info('Total params: {:,}'.format(trainable_count + non_trainable_count))
    logging_msg.info('Trainable params: {:,}'.format(trainable_count))
    logging_msg.info('Non-trainable params: {:,}'.format(non_trainable_count))
