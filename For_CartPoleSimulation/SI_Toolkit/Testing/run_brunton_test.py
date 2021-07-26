# "Command line" parameters
from SI_Toolkit.Testing.Parameters_for_testing import args

from SI_Toolkit.Testing.Testing_Functions.preprocess_for_brunton import preprocess_for_brunton
from SI_Toolkit.Testing.Testing_Functions.get_prediction_TF_predictor import get_data_for_gui_TF
from SNN.get_prediction_SNN_predictor import get_data_for_gui_SNN
from SI_Toolkit.Testing.Testing_Functions.get_prediction_from_euler_predictor import get_prediction_from_euler_predictor
from SI_Toolkit.Testing.Testing_Functions.Brunton_GUI import run_test_gui

import yaml, os

print('')
a = args()  # 'a' like arguments
print(a.__dict__)
print('')

def run_brunton_test():

    dataset, time_axis, dataset_sampling_dt, ground_truth = preprocess_for_brunton(a)
    config = yaml.load(open(os.path.join('SI_Toolkit_ApplicationSpecificFiles', 'config.yml'), 'r'),
                       Loader=yaml.FullLoader)

    NET_TYPE = config['modeling']['NET_TYPE']
    BRUNTON_MODE = config['testing']['BRUNTON_MODE']

    predictions_list = []
    for test_idx in range(len(a.tests)):
        if a.tests[test_idx] == 'Euler':
            predictions = get_prediction_from_euler_predictor(a, dataset, dataset_sampling_dt, dt_sampling_by_dt_fine=10)
        else:  # Assume this is a neural_network test:
            if (NET_TYPE == 'SNN'):
                print('SNN yeah!')
                predictions = get_data_for_gui_SNN(a, dataset, net_name=a.tests[test_idx], mode=BRUNTON_MODE)
            else:
                predictions = get_data_for_gui_TF(a, dataset, net_name=a.tests[test_idx])


        #print(predictions)

        predictions_list.append(predictions)

    run_test_gui(a.features, a.titles,
                 ground_truth, predictions_list, time_axis,
                 )


if __name__ == '__main__':
    run_brunton_test()
