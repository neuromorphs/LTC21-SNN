import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#mpl.use("Qt5Agg")
from tqdm import tqdm
from pathlib import Path
import datetime
from models.predictive_model import PredictiveModelLMU
from plot_predictions import plot_state_prediction, plot_error_curve
from utils.data import load_datasets, get_scaling_factor, scale_datasets
import pickle
import time
# import nengo_ocl

# setup some parameters
# TODO these should come from an argparser or config file!
model_name = "LMU_model"
experiment_name = "training"
data_dir = "data/Train/"
results_dir = "results/"
load_state = ""
action_vars = ["Q"]
state_vars = ["angle_sin", "angle_cos", "angleD", "position", "positionD"]
bound = 0.19  # if the cart ever leaves these bounds, the data is ignored
epochs = 1  # number or epochs for training
sample_freq = 50  # cartpole data is recorded at ~50Hz
dt = 0.001  # nengo time step
learning_rate = 5e-5  # lr
t_delays = [0.02, 0.06, 0.1]  # how far to predict the future (initial guess)
neurons_per_dim = 50  # number of neurons representing each dimension
seed = 4  # to get reproducible neuron properties across runs
lmu_theta = 0.1  # duration of the LMU delay
lmu_q = 5  # number of factorizations per dim in LMU
max_samples = -1  # just reduce training set for quick debugging
plot_prediction_every = 100  # how often to plot a prediction curve during learning

# crating a unique folder to save the weights in
folder_name = (
    str(datetime.datetime.now().date())
    + "_"
    + str(datetime.datetime.now().time()).replace(":", ".")
)
run_dir = Path(results_dir, experiment_name, model_name, folder_name)
run_dir.mkdir(parents=True, exist_ok=True)

# load training data from disk
training_data = load_datasets(data_dir, bound=bound)
print(f"training data contains {len(training_data)} files")

# retreive the scaling factor
scales = get_scaling_factor(training_data, state_vars)
print("detected scaling factors:")
for k, v in scales.items():
    print(f"{k:15s}  :  {v:3.3f}")

# scale datasets to [-1,1]
training_data = scale_datasets(training_data, scales)

# init weights from file or empty
if load_state:
    print("loading model state from", load_state)
    with open(load_state, "rb") as f:
        model_state = pickle.load(f)
    model = PredictiveModelLMU(**model_state)
else:
    print("initializing weights as zeros")
    weights = None  # let the model initialize its own weights
    # init the model object
    model = PredictiveModelLMU(
        seed=seed, neurons_per_dim=neurons_per_dim, sample_freq=sample_freq,
        lmu_theta=lmu_theta, lmu_q=lmu_q, radius=1.5, dt=dt,
        t_delays=t_delays, learning_rate=learning_rate, action_vars=action_vars,
        state_vars=state_vars, weights=weights, scales=scales
    )

# get the model weights
weights = model.get_weights()

# record the training error over time
all_prediction_errors = [[] for _ in t_delays]
all_baseline_errors = [[] for _ in t_delays]
all_extra_errors = [[] for _ in t_delays]

for e in range(epochs):
    print("\nstarting epoch", e + 1)
    start_time = time.time()

    # record the training error for the epoch
    epoch_mean_prediction_errors = [[] for _ in t_delays]
    epoch_mean_baseline_errors = [[] for _ in t_delays]
    epoch_mean_extra_errors = [[] for _ in t_delays]

    # loop over the training dataset
    with tqdm(total=len(training_data[:max_samples])) as t:
        for i, df in enumerate(training_data[:max_samples]):

            # reset the model neurons
            model.reset_sim()

            # pass the training data
            action_df = df[["time"] + action_vars]
            state_df = df[["time"] + state_vars]
            model.set_inputs(action_df, state_df)

            # set the correct model weights
            model.set_weights(weights)

            # run the simulation
            recordings = model.process_files()

            # collect the output data
            actions = model.sim.data[recordings["actions"]]
            states = model.sim.data[recordings["states"]]
            weights = model.get_weights()

            # TODO: THE WEIGHTS ARE LOOKING WEIRD, WHY?
            # print(weights)

            for j, t_d in enumerate(t_delays):

                # retrieve network output based on delay
                predicted_states = model.sim.data[recordings["predictions"][f"{j}"]["states_pred"]]
                prediction_errors = model.sim.data[recordings["predictions"][f"{j}"]["errors"]]

                # report the prediction error (next state - predicted next state)
                mean_prediction_error = np.mean(np.abs(prediction_errors))
                epoch_mean_prediction_errors[j].append(mean_prediction_error)
                all_prediction_errors[j].append(mean_prediction_error)

                delta_t = int(t_d / dt)  # number of network simulated timesteps for lookahead
                # report the difference between current state and next state
                mean_baseline_error = np.mean(np.abs(states[:-delta_t] - states[delta_t:]))
                epoch_mean_baseline_errors[j].append(mean_baseline_error)
                all_baseline_errors[j].append(mean_baseline_error)

                # report the difference between prediction and linear extrapolation
                p_s_extrapolation = 2 * states[delta_t:-delta_t] - states[:-2*delta_t]
                mean_extrapolation_error = np.mean(np.abs(p_s_extrapolation - states[2*delta_t:]))
                epoch_mean_extra_errors[j].append(mean_extrapolation_error)
                all_extra_errors[j].append(mean_extrapolation_error)

                # plot the prediction
                if (i % plot_prediction_every) == 0:
                    fig = plot_state_prediction(
                        states,
                        predicted_states,
                        p_extra=p_s_extrapolation,
                        delta_t=delta_t,
                        state_vars=state_vars,
                        save_path=Path(run_dir, f"prediction_e{e}_i{i}_td{delta_t}_training.svg"),
                        show=True
                    )
                    plt.close()

            # update the loading bar
            t.set_postfix(loss="{:05.4f}".format(mean_prediction_error))
            t.update()

    runtime = time.time() - start_time
    print(f"epoch processing time: {int(runtime//60)}m {runtime%60:.3f}s")

    for j, t_d in enumerate(t_delays):

        # save the weights
        model.save_state_dict(Path(run_dir, "model_state.pkl"))

        # report epoch errors
        print()
        print(f"prediction delay = {t_d}")
        print()
        print(f"epoch mean prediction error   : {np.mean(epoch_mean_prediction_errors[j])}")
        print(f"epoch mean baseline error     : {np.mean(epoch_mean_baseline_errors[j])}")
        print(f"epoch mean extrapolation error: {np.mean(epoch_mean_extra_errors[j])}")

        fig = plot_error_curve(
            all_prediction_errors[j],
            all_baseline_errors[j],
            all_extra_errors[j],
            t_delay=t_d,
            save_path=Path(run_dir, f"error_curve_td{t_d}.svg"),
            show=True
        )
        plt.close()