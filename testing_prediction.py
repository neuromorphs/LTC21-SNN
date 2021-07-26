import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#mpl.use("Qt5Agg")
from tqdm import tqdm
from pathlib import Path
import datetime
from models.predictive_model import PredictiveModelAutoregressiveLMU
from plot_predictions import plot_state_prediction, plot_error_curve
from utils.data import load_datasets, scale_datasets
from utils.config_loader import load_config
import pickle
import time

# load the training parameters from config file
config = load_config("test_config.yaml")

"""
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
"""

# load parameters from state dict
print("loading model state from", config["load_state"])
with open(config["load_state"], "rb") as f:
    model_state = pickle.load(f)

# set the random seed for numpy
np.random.seed(model_state["seed"])

# set the learning rate to 0 during testing
model_state["learning_rate"] = 0.0
model_state["t_switch"] = config["t_switch"]

# init the model
model = PredictiveModelAutoregressiveLMU(**model_state)

# load training data from disk
test_data = load_datasets(
    config["data_dir"],
    bound=config["bound"],
    nrows=config["datapoints_per_file"],
    shuffle=False,
    skiprows=config["skiprows"]
)
print(f"test data contains {len(test_data)} files")

# scale datasets to [-1,1]
print("detected scaling factors:")
for k, v in model_state["scales"].items():
    print(f"{k:15s}  :  {v:3.3f}")
test_data = scale_datasets(test_data, model_state["scales"])

# initialize the model with loaded parameters
model = PredictiveModelAutoregressiveLMU(**model_state)

# crating a unique folder to save the weights in
folder_name = (
    str(datetime.datetime.now().date())
    + "_"
    + str(datetime.datetime.now().time()).replace(":", ".")
)
run_dir = Path(config["results_dir"], config["experiment_name"], model_state["model_name"], folder_name)
run_dir.mkdir(parents=True, exist_ok=True)

# record the training error over time
all_prediction_errors = []
all_baseline_errors = []
all_extra_errors = []

# number of network simulated timesteps for lookahead
delta_t = int(model_state["t_delay"] / model_state["dt"])

start_time = time.time()

# loop over the training dataset
with tqdm(total=len(test_data[:config["max_samples"]])) as t:
    for i, df in enumerate(test_data[:config["max_samples"]]):

        # reset the model neurons
        model.reset_sim()

        # pass the test data
        action_df = df[["time"] + model_state["action_vars"]]
        state_df = df[["time"] + model_state["state_vars"]]
        model.set_inputs(action_df, state_df)

        # run the simulation
        recordings = model.process_files()

        # collect the output data
        actions = model.sim.data[recordings["actions"]]
        states = model.sim.data[recordings["states"]]
        # predicted_current_states = model.sim.data[recordings["predicted_current_states"]]
        predicted_future_states = model.sim.data[recordings["predicted_future_states"]]
        prediction_errors = model.sim.data[recordings["prediction_errors"]]

        # report the prediction error (next state - predicted next state)
        # mean_prediction_error = np.mean(np.abs(prediction_errors))
        mean_prediction_error = np.mean(np.abs(predicted_future_states[delta_t:-delta_t] - states[2 * delta_t:]))
        all_prediction_errors.append(mean_prediction_error)

        # report the difference between current state and next state
        mean_baseline_error = np.mean(np.abs(states[:-delta_t] - states[delta_t:]))
        all_baseline_errors.append(mean_baseline_error)

        # report the difference between prediction and linear extrapolation
        p_s_extrapolation = 2 * states[delta_t:-delta_t] - states[:-2 * delta_t]
        mean_extrapolation_error = np.mean(np.abs(p_s_extrapolation - states[2 * delta_t:]))
        all_extra_errors.append(mean_extrapolation_error)

        # plot the prediction
        if (i % config["plot_prediction_every"]) == 0:
            fig = plot_state_prediction(
                states,
                predicted_future_states,
                p_extra=p_s_extrapolation,
                delta_t=delta_t,
                state_vars=model_state["state_vars"],
                save_path=Path(run_dir, f"prediction_i{i}_td{delta_t}_testing.svg"),
                show=True
            )
            plt.close()

            plt.plot(prediction_errors)
            plt.title(f"prediction error model dt = {model_state['t_delay']}")
            plt.show()
            plt.close()

            plt.plot(predicted_future_states[:-delta_t] - states[delta_t:])
            plt.title(f"prediction error after dt = {model_state['t_delay']}")
            plt.show()
            plt.close()

        # update the loading bar
        t.set_postfix(loss="{:05.4f}".format(mean_prediction_error))
        t.update()

runtime = time.time() - start_time
print(f"test data processing time: {int(runtime//60)}m {runtime%60:.3f}s")

# report test errors
print()
print(f"prediction delay = {model_state['t_delay']}")
print()
print(f"mean prediction error   : {np.mean(all_prediction_errors):.4f}")
print(f"mean baseline error     : {np.mean(all_baseline_errors):.4f}")
print(f"mean extrapolation error: {np.mean(all_extra_errors):.4f}")

fig = plot_error_curve(
    all_prediction_errors,
    all_baseline_errors,
    all_extra_errors,
    t_delay=model_state['t_delay'],
    save_path=Path(run_dir, f"error_curve_td{model_state['t_delay']}.svg"),
    show=True
)
plt.close()