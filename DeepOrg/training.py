from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

from DeepOrg.models import model_logger


def train(model, model_name, optimizer, loss, save_file_name, data_generator_train, data_generator_val,
          total_train_size, total_val_size, nb_epoch, batch_size, logging, logging_msg):

    model_logger.log_parameter_numbers(model, model_name, logging_msg)

    # we need to recompile the model for these modifications to take effect
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10)
    csv_logger = CSVLogger(save_file_name + '.csv')
    model_checkpoint = ModelCheckpoint(save_file_name + '.hdf5', monitor='val_loss', save_best_only=True)

    logging.info('################################')
    if 'top' in save_file_name:
        logging.info('top training starts...')
    else:
        logging.info('training starts...')
    logging.info('################################')

    history = model.fit_generator(data_generator_train, steps_per_epoch=total_train_size // batch_size, epochs=nb_epoch, verbose=1,
                                  validation_data=data_generator_val, validation_steps=total_val_size // batch_size,
                                  callbacks=[csv_logger, early_stopper, model_checkpoint])

    return history
