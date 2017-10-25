import os

from DeepOrg.tools import alerts


def init(save_result_path, save_file_name):
    if not os.path.exists(save_result_path):
        os.mkdir(save_result_path)

    if os.path.exists(save_file_name + '_log.txt'):
        choice = alerts.show_delete(
            'Log file exists for this model. hdf5 and others may also exist. Do you want to delete previous ones?'
        )
        if choice:
            # delete all previous files
            # make sure all model names are inside '()', as unique IDs
            for file in os.listdir(save_result_path):
                if file.split(')')[0] == os.path.basename(save_file_name).split(')')[0]:
                    os.remove(os.path.join(save_result_path, file))
        else:
            exit(0)
