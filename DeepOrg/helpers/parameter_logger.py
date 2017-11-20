def log_info(nb_classes, img_rows, img_cols, img_channels, batch_size, nb_epoch,
             steps_train, steps_val, total_test_size, logging_msg, nb_epoch_top=None):
    logging_msg.info('############################################################################')
    logging_msg.info('nb_classes: %d \t image info: %dX%d, %d' % (nb_classes, img_rows, img_cols, img_channels))
    if nb_epoch_top is None:
        logging_msg.info('batch_size: %d \t nb_epoch: %d' % (batch_size, nb_epoch))
    else:
        logging_msg.info('batch_size: %d \t nb_epoch_top: %d \t nb_epoch: %d' % (batch_size, nb_epoch_top, nb_epoch))
    logging_msg.info('total train size: %d \t total val size: %d \t total test size: %d' % (steps_train * batch_size, steps_val * batch_size, total_test_size))
    logging_msg.info('############################################################################')
