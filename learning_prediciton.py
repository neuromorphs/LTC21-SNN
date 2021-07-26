import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#mpl.use("Qt5Agg")
from tqdm import tqdm
from pathlib import Path
import datetime
from models.predictive_model import PredictiveModelAutoregressiveLMU
from plot_predictions import plot_state_prediction, plot_error_curve
from utils.data import load_datasets, get_scaling_factor, scale_datasets
from utils.config_loader import load_config
import pickle
import time

# load the training parameters from config file
config = load_config("training_config.yaml")

# set the random seed for numpy
np.random.seed(config["seed"])

# load training data from disk
training_data = load_datasets(
    config["data_dir"],
    bound=config["bound"],
    nrows=config["datapoints_per_file"],
    shuffle=config["shuffle"],
    skiprows=config["skiprows"]
)
print(f"training data contains {len(training_data)} files")

# retreive the scaling factor
scales = get_scaling_factor(training_data, config["state_vars"])
print("detected scaling factors:")
for k, v in scales.items():
    print(f"{k:15s}  :  {v:3.3f}")

# scale datasets to [-1,1]
training_data = scale_datasets(training_data, scales)

# init weights from file or empty
if config["load_state"]:
    print("loading model state from", config["load_state"])
    with open(config["load_state"], "rb") as f:
        model_state = pickle.load(f)
    model = PredictiveModelAutoregressiveLMU(**model_state)
else:
    # init the model object with zero weights
    print("initializing weights as zeros")
    model = PredictiveModelAutoregressiveLMU(scales=scales, **config)

# crating a unique folder to save the weights in
folder_name = (
    str(datetime.datetime.now().date())
    + "_"
    + str(datetime.datetime.now().time()).replace(":", ".")
)
run_dir = Path(config["results_dir"], config["experiment_name"], config["model_name"], folder_name)
run_dir.mkdir(parents=True, exist_ok=True)

# get the model weights
weights = model.get_weights()

# record the training error over time
all_prediction_errors = []
all_baseline_errors = []
all_extra_errors = []

# number of network simulated timesteps for lookahead
delta_t = int(config["t_delay"] / config["dt"])

for e in range(1, config["epochs"]+1):
    print("\nstarting epoch", e)
    start_time = time.time()

    # record the training error for the epoch
    epoch_mean_prediction_errors = []
    epoch_mean_baseline_errors = []
    epoch_mean_extra_errors = []

    # loop over the training dataset
    with tqdm(total=len(training_data[:config["max_samples"]])) as t:
        for i, df in enumerate(training_data[:config["max_samples"]]):

            # reset the model neurons
            model.reset_sim()

            # pass the training data
            action_df = df[["time"] + config["action_vars"]]
            state_df = df[["time"] + config["state_vars"]]
            model.set_inputs(action_df, state_df)

            # set the correct model weights
            model.set_weights(weights)

            # run the simulation
            recordings = model.process_files()

            # collect the output data
            actions = model.sim.data[recordings["actions"]]
            states = model.sim.data[recordings["states"]]
            # predicted_current_states = model.sim.data[recordings["predicted_current_states"]]
            predicted_future_states = model.sim.data[recordings["predicted_future_states"]]
            prediction_errors = model.sim.data[recordings["prediction_errors"]]

            weights = model.get_weights()

            # TODO: THE WEIGHTS ARE LOOKING WEIRD, WHY?
            # print(weights)

            # report the prediction error (next state - predicted next state)
            #mean_prediction_error = np.mean(np.abs(prediction_errors))
            mean_prediction_error = np.mean(np.abs(predicted_future_states[delta_t:-delta_t] - states[2*delta_t:]))
            epoch_mean_prediction_errors.append(mean_prediction_error)
            all_prediction_errors.append(mean_prediction_error)

            # report the difference between current state and next state
            mean_baseline_error = np.mean(np.abs(states[:-delta_t] - states[delta_t:]))
            epoch_mean_baseline_errors.append(mean_baseline_error)
            all_baseline_errors.append(mean_baseline_error)

            # report the difference between prediction and linear extrapolation
            p_s_extrapolation = 2 * states[delta_t:-delta_t] - states[:-2*delta_t]
            mean_extrapolation_error = np.mean(np.abs(p_s_extrapolation - states[2*delta_t:]))
            epoch_mean_extra_errors.append(mean_extrapolation_error)
            all_extra_errors.append(mean_extrapolation_error)

            # plot the prediction
            if (i % config["plot_prediction_every"]) == 0:
                fig = plot_state_prediction(
                    states,
                    predicted_future_states,
                    p_extra=p_s_extrapolation,
                    delta_t=delta_t,
                    state_vars=config["state_vars"],
                    save_path=Path(run_dir, f"prediction_e{e}_i{i}_td{delta_t}_training.svg"),
                    show=True
                )
                plt.close()

                plt.plot(prediction_errors)
                plt.title(f"prediction error model dt = {config['t_delay']}")
                plt.show()
                plt.close()

                plt.plot(predicted_future_states[:-delta_t] - states[delta_t:])
                plt.title(f"prediction error after dt = {config['t_delay']}")
                plt.show()
                plt.close()

            # save the weights
            if (i % config["save_state_every"]) == 0:
                model.save_state_dict(Path(run_dir, "model_state.pkl"))

            # update the loading bar
            t.set_postfix(loss="{:05.4f}".format(mean_prediction_error))
            t.update()

    runtime = time.time() - start_time
    print(f"epoch processing time: {int(runtime//60)}m {runtime%60:.3f}s")

    # save the weights
    model.save_state_dict(Path(run_dir, "model_state.pkl"))

    # report epoch errors
    print()
    print(f"prediction delay = {config['t_delay']}")
    print()
    print(f"epoch mean prediction error   : {np.mean(epoch_mean_prediction_errors):.4f}")
    print(f"epoch mean baseline error     : {np.mean(epoch_mean_baseline_errors):.4f}")
    print(f"epoch mean extrapolation error: {np.mean(epoch_mean_extra_errors):.4f}")

    fig = plot_error_curve(
        all_prediction_errors,
        all_baseline_errors,
        all_extra_errors,
        t_delay=config['t_delay'],
        save_path=Path(run_dir, f"error_curve_td{config['t_delay']}.svg"),
        show=True
    )
    plt.close()