import os

from tools import alerts


def init(save_result_path, model_name):
    if not os.path.exists(save_result_path):
        os.mkdir(save_result_path)

    if os.path.exists(os.path.join(save_result_path, model_name + '_log.txt')):
        choice = alerts.show_delete(
            'Log file exists for this model. hdf5 and others may also exist. Do you want to delete previous ones?'
        )
        if choice:
            # delete all previous files
            for file in os.listdir(save_result_path):
                if file.startswith(model_name):
                    os.remove(os.path.join(save_result_path, file))
        else:
            exit(0)
