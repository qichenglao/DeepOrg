import numpy as np

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

from DeepOrg.models import model_logger


def train(model, model_name, save_file_name, logging, logging_msg, steps_train, steps_val,
          data_generator_train, data_generator_val, batch_size, nb_epoch, loss, optimizer, early_stop, loss_weights=None):

    model_logger.log_parameter_numbers(model, model_name, logging_msg)

    # we need to recompile the model for these modifications to take effect
    model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer, metrics=['accuracy'])

    early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10)
    csv_logger = CSVLogger(save_file_name + '.csv')
    model_checkpoint = ModelCheckpoint(save_file_name + '.hdf5', monitor='val_loss', save_best_only=True)

    logging.info('################################')
    if 'top' in save_file_name:
        logging.info('top training starts...')
    else:
        logging.info('training starts...')
    logging.info('################################')

    # steps_train = int(np.ceil(total_train_size / float(batch_size)))
    # steps_val = int(np.ceil(total_val_size / float(batch_size)))

    if early_stop:
        history = model.fit_generator(data_generator_train, steps_per_epoch=steps_train, epochs=nb_epoch, verbose=1,
                                      validation_data=data_generator_val, validation_steps=steps_val,
                                      callbacks=[csv_logger, early_stopper, model_checkpoint])
    else:
        history = model.fit_generator(data_generator_train, steps_per_epoch=steps_train, epochs=nb_epoch, verbose=1,
                                      validation_data=data_generator_val, validation_steps=steps_val,
                                      callbacks=[csv_logger, model_checkpoint])

    return history
