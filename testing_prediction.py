import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#mpl.use("Qt5Agg")
from tqdm import tqdm
from pathlib import Path
import datetime
from models.predictive_model import PredictiveModelLMU
from plot_predictions import plot_state_prediction, plot_error_curve
from utils.data import load_datasets, scale_datasets
import pickle
import time

# setup some parameters
# TODO these should come from an argparser or config file!
model_name = "LMU_model"
experiment_name = "testing"
data_dir = "data/Validate/"
results_dir = "results/"
load_state = "results/test1/LMU_model/2021-07-12_14.28.08.184663/model_state.pkl"
bound = 0.19  # if the cart ever leaves these bounds, the data is ignored
plot_prediction_every = 1  # how often to plot a prediction curve during testing
max_samples = -1  # just reduce test set for quick debugging (-1 = use all data)

# crating a unique folder to save the weights in
folder_name = (
    str(datetime.datetime.now().date())
    + "_"
    + str(datetime.datetime.now().time()).replace(":", ".")
)
run_dir = Path(results_dir, experiment_name, model_name, folder_name)
run_dir.mkdir(parents=True, exist_ok=True)

# load training data from disk
test_data = load_datasets(data_dir, bound=bound)
print(f"training data contains {len(test_data)} files")

# load parameters from state dict
print("loading model state from", load_state)
with open(load_state, "rb") as f:
    model_state = pickle.load(f)
scales = model_state["scales"]
t_delays = model_state["t_delays"]
action_vars = model_state["action_vars"]
state_vars = model_state["state_vars"]
dt = model_state["dt"]

# set the learning rate to 0 during testing
model_state["learning_rate"] = 0.0

# scale datasets to [-1,1]
print("detected scaling factors:")
for k, v in scales.items():
    print(f"{k:15s}  :  {v:3.3f}")
test_data = scale_datasets(test_data, scales)

# initialize the model with loaded parameters
model = PredictiveModelLMU(**model_state)

# record the training error over time
all_mean_prediction_errors = [[] for _ in t_delays]
all_mean_baseline_errors = [[] for _ in t_delays]
all_mean_extra_errors = [[] for _ in t_delays]

start_time = time.time()

# loop over the training dataset
with tqdm(total=len(test_data[:max_samples])) as t:
    for i, df in enumerate(test_data[:max_samples]):

        # reset the model neurons
        model.reset_sim()

        # pass the training data
        action_df = df[["time"] + action_vars]
        state_df = df[["time"] + state_vars]
        model.set_inputs(action_df, state_df)

        # run the simulation
        recordings = model.process_files()

        # collect the output data
        actions = model.sim.data[recordings["actions"]]
        states = model.sim.data[recordings["states"]]

        for j, t_d in enumerate(t_delays):

            # retrieve network output based on delay
            predicted_states = model.sim.data[recordings["predictions"][f"{j}"]["states_pred"]]
            prediction_errors = model.sim.data[recordings["predictions"][f"{j}"]["errors"]]

            # report the prediction error (next state - predicted next state)
            mean_prediction_error = np.mean(np.abs(prediction_errors))
            all_mean_prediction_errors[j].append(mean_prediction_error)

            delta_t = int(t_d / dt)  # number of network simulated timesteps for lookahead
            # report the difference between current state and next state
            mean_baseline_error = np.mean(np.abs(states[:-delta_t] - states[delta_t:]))
            all_mean_baseline_errors[j].append(mean_baseline_error)

            # report the difference between prediction and linear extrapolation
            p_s_extrapolation = 2 * states[delta_t:-delta_t] - states[:-2*delta_t]
            mean_extrapolation_error = np.mean(np.abs(p_s_extrapolation - states[2*delta_t:]))
            all_mean_extra_errors[j].append(mean_extrapolation_error)

            # plot the prediction
            if (i % plot_prediction_every) == 0:
                fig = plot_state_prediction(
                    states,
                    predicted_states,
                    p_extra=p_s_extrapolation,
                    delta_t=delta_t,
                    state_vars=state_vars,
                    save_path=Path(run_dir, f"prediction_i{i}_td{delta_t}_testing.svg"),
                    show=True
                )
                plt.close()

        # update the loading bar
        t.set_postfix(loss="{:05.4f}".format(mean_prediction_error))
        t.update()

runtime = time.time() - start_time
print(f"test data processing time: {int(runtime//60)}m {runtime%60:.3f}s")

for j, t_d in enumerate(t_delays):

    # report epoch errors
    print()
    print(f"prediction delay = {t_d}")
    print()
    print(f"mean prediction error   : {np.mean(all_mean_prediction_errors[j])}")
    print(f"mean baseline error     : {np.mean(all_mean_baseline_errors[j])}")
    print(f"mean extrapolation error: {np.mean(all_mean_extra_errors[j])}")

    fig = plot_error_curve(
        all_mean_prediction_errors[j],
        all_mean_baseline_errors[j],
        all_mean_extra_errors[j],
        t_delay=t_d,
        save_path=Path(run_dir, f"error_curve_td{t_d}.svg"),
        show=True
    )
    plt.close()