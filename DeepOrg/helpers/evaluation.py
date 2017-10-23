from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix


def evaluate(X_test, y_test, nb_classes, model, logging, logging_msg):
    # pred = np.empty(y_test.shape[0], dtype="uint32")
    pred = model.predict(X_test).argmax(axis=-1)
    # pred = model.predict_generator(data_test_generator(), steps=X_test.shape[0] // batch_size)

    logging.info(pred)
    logging.info(y_test)
    logging.info(y_test == pred)

    # Convert class vectors to binary class matrices.
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    PRED = np_utils.to_categorical(pred, nb_classes)

    score = model.evaluate(X_test, Y_test)
    # score = model.evaluate_generator(data_test_generator(), steps=X_test.shape[0] // batch_size)
    logging.info(score)
    logging_msg.info(classification_report(Y_test, PRED))
    logging_msg.info(confusion_matrix(y_test, pred))
