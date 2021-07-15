import numpy as np
from matplotlib import colors

from tqdm import trange

try:
    from SI_Toolkit_ApplicationSpecificFiles.predictors_customization import STATE_VARIABLES, STATE_INDICES
except ModuleNotFoundError:
    print('SI_Toolkit_ApplicationSpecificFiles not yet created')

from SNN.predictor_autoregressive_SNN import predictor_autoregressive_SNN

# This import mus go before pyplot so also before our scripts
from matplotlib import use, get_backend
# Use Agg if not in scientific mode of Pycharm

if get_backend() != 'module://backend_interagg':
    use('Agg')


cdict = {'red':   ((0.0,  0.22, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.89, 1.0)),

         'green': ((0.0,  0.49, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.12, 1.0)),

         'blue':  ((0.0,  0.72, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  0.11, 1.0))}

cmap = colors.LinearSegmentedColormap('custom', cdict)

def get_data_for_gui_SNN(a, dataset, net_name, mode):
    states_0 = dataset[STATE_VARIABLES].to_numpy()[:-a.test_max_horizon, :]

    Q = dataset['Q'].to_numpy()
    Q_array = [Q[i:-a.test_max_horizon+i] for i in range(a.test_max_horizon)]
    Q_array = np.vstack(Q_array).transpose()

    if mode == 'batch':
        print('batch mode')
        # All at once
        predictor = predictor_autoregressive_SNN(horizon=a.test_max_horizon, batch_size=a.test_len, net_name=net_name)
        predictor.setup(initial_state=states_0, prediction_denorm=True)
        output_array = predictor.predict(Q_array)
    elif mode == 'sequential':
        print('sequential mode')
        predictor = predictor_autoregressive_SNN(horizon=a.test_max_horizon, batch_size=1, net_name=net_name)

        # Iteratively (to test internal state update)
        output_array = np.zeros([a.test_len, a.test_max_horizon + 1, len(STATE_VARIABLES) + 1], dtype=np.float32)
        for timestep in trange(a.test_len):
            Q_current_timestep = Q_array[np.newaxis, timestep, :]
            s_current_timestep = states_0[timestep, np.newaxis]

            predictor.setup(initial_state=s_current_timestep, prediction_denorm=True)
            #predictor.setup(initial_state=s_current_timestep, prediction_denorm=False)

            output_array[timestep,:,:] = predictor.predict(Q_current_timestep)
            predictor.update_internal_state(Q_current_timestep[0, 0])

    else:
        raise ValueError("Mode should be either 'batch' or 'sequential'")

    output_array = output_array[..., [STATE_INDICES.get(key) for key in a.features]+[-1]]
    #print(output_array.shape) # shape = (161,41,7)
    #print(output_array)

    # time_axis is a time axis for ground truth
    return output_array
