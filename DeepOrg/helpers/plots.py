import matplotlib.pyplot as plt
import os


def save_training_plots(history, save_file_name):
    # list all data in history
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.savefig(save_file_name + '_model_accuracy.jpg')
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(save_file_name + '_model_loss.jpg')
    plt.close()


def save_training_plots_by_key(history, save_file_name, train_key, val_key):
    print(train_key, val_key)

    plt.plot(history.history[train_key])
    plt.plot(history.history[val_key])
    plt.title('model ' + train_key)
    plt.ylabel(train_key)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.savefig(save_file_name + '_model_' + train_key + '.jpg')
    plt.close()
