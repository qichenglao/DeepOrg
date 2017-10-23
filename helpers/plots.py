import matplotlib.pyplot as plt
import os


def save_training_plots(history, fileID, nfold, save_path):
    # list all data in history
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy - fold ' + str(nfold))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.savefig(os.path.join(save_path, fileID + '_model_accuracy_fold_' + str(nfold) + '.jpg'))
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss - fold ' + str(nfold))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(os.path.join(save_path, fileID + '_model_loss_fold_' + str(nfold) + '.jpg'))
    plt.close()
