import pandas as pd
import numpy as np
import nengo
import matplotlib as mpl
import matplotlib.pyplot as plt
#mpl.use("Qt5Agg")
import os
from tqdm import tqdm
from pathlib import Path
import datetime
from models.predictive_model import make_model, make_model_LMU, make_model_LMU2, make_model_LMU3
from plot_predictions import plot_state_prediction, plot_error_curve
from utils.data import load_datasets, scale_datasets

model_name = "LMU3"
experiment_name = "test1"
data_dir = "data/Train/"
results_dir = "results/"
load_weights = ""
bound = 0.19  # if the cart ever leaves these bounds, the data is ignored
epochs = 10  # number or epochs for training
samp_freq = 50  # cartpole data is recorded at ~50Hz
dt = 0.001  # nengo time step
learning_rate = 5e-5  # lr
t_delay = 0.02  # how far to predict the future (initial guess)
neurons_per_dim = 50  # number of neurons representing each dimension
seed = 4  # to get reproducible neuron properties across runs
lmu_theta = 0.1  # duration of the LMU delay
lmu_q = 5  # number of factorizations per dim in LMU

# crating a unique folder to save the weights in
folder_name = (
    str(datetime.datetime.now().date())
    + "_"
    + str(datetime.datetime.now().time()).replace(":", ".")
)
run_dir = Path(results_dir, experiment_name, model_name, folder_name)
run_dir.mkdir(parents=True, exist_ok=True)

P_A = 0
P_S = 1
P_Z = 2
P_Z_PRED = 3
P_E = 4
P_WEIGHTS = 5

# load training data from disk
training_data = load_datasets(data_dir)
print(f"training data contains {len(training_data)} files")

# scale datasets to [-1,1]
training_data = scale_datasets(training_data)

# init weights from file or empty
if load_weights:
    print("loading weights from", load_weights)
    weights = np.load(load_weights)
else:
    print("initializing weights as zeros")
    weights = None  # let the model initialize its own weights

# train the model
all_prediction_errors = []
all_baseline_errors = []
all_extra_errors = []
for e in range(epochs):
    print("\nstarting epoch", e + 1)
    epoch_mean_prediction_errors = []
    epoch_mean_baseline_errors = []
    epoch_mean_extra_errors = []
    with tqdm(total=len(training_data)) as t:
        for i, df in enumerate(training_data):
            action_df = df[["time", "Q"]]
            state_df = df[
                [
                    "time",
                    "angle",
                    "angleD",
                    # "angleDD",
                    "angle_cos",
                    "angle_sin",
                    "position",
                    "positionD",
                    # "positionDD",
                    # "target_position",
                ]
            ]
            t_max = action_df["time"].max()  # number of seconds to run

            if model_name == "LMU":
                model, recordings = make_model_LMU(
                    action_df,
                    state_df,
                    weights=weights,
                    seed=seed,
                    n=neurons_per_dim,
                    samp_freq=samp_freq,
                    t_delay=t_delay,
                    lmu_theta=lmu_theta,
                    lmu_q=lmu_q,
                    learning_rate=learning_rate,
                )
            elif model_name == "LMU2":
                model, recordings = make_model_LMU2(
                    action_df,
                    state_df,
                    weights=weights,
                    seed=seed,
                    n=neurons_per_dim,
                    samp_freq=samp_freq,
                    t_delay=t_delay,
                    lmu_theta=lmu_theta,
                    lmu_q=lmu_q,
                    learning_rate=learning_rate,
                )
            elif model_name == "LMU3":
                model, recordings = make_model_LMU3(
                    action_df,
                    state_df,
                    weights=weights,
                    seed=seed,
                    n=neurons_per_dim,
                    samp_freq=samp_freq,
                    t_delay=t_delay,
                    lmu_theta=lmu_theta,
                    lmu_q=lmu_q,
                    learning_rate=learning_rate,
                )
            else:
                model, recordings = make_model(
                    action_df,
                    state_df,
                    weights=weights,
                    seed=seed,
                    n=neurons_per_dim,
                    samp_freq=samp_freq,
                    t_delay=t_delay,
                    learning_rate=learning_rate,
                )

            # run the simulation
            sim = nengo.Simulator(model, progress_bar=False)
            sim.run(t_max)

            # collect the output data
            weights = sim.data[recordings[P_WEIGHTS]][-1]
            p_e = sim.data[recordings[P_E]]
            p_z = sim.data[recordings[P_Z]]
            p_z_pred = sim.data[recordings[P_Z_PRED]]
            p_s = sim.data[recordings[P_S]]

            # report the prediction error
            mean_prediction_error = np.mean(np.abs(p_e))
            epoch_mean_prediction_errors.append(mean_prediction_error)
            all_prediction_errors.append(mean_prediction_error)

            # report the difference between prediction and last state
            mean_baseline_error = np.mean(np.abs(p_z_pred - p_s))
            epoch_mean_baseline_errors.append(mean_baseline_error)
            all_baseline_errors.append(mean_baseline_error)

            # report the difference between prediction and linear extrapolation
            delta_t = int(t_delay / dt)
            p_s_extrapolation = 2 * p_s[delta_t:-delta_t] - p_s[:-2*delta_t]
            mean_extrapolation_error = np.mean(np.abs(p_s_extrapolation - p_z[2*delta_t:]))
            epoch_mean_extra_errors.append(mean_extrapolation_error)
            all_extra_errors.append(mean_extrapolation_error)

            # update the loading bar
            t.set_postfix(loss="{:05.4f}".format(mean_prediction_error))
            t.update()

            # plot the prediction
            if (i+1) % 100 == 0:
                fig = plot_state_prediction(
                    p_z,
                    p_z_pred,
                    p_extra=p_s_extrapolation,
                    delta_t=delta_t,
                    save_path=Path(run_dir, f"prediction_e{e}_i{i}_training.svg"),
                    show=True
                )
                plt.close()

    # save the weights
    np.save(Path(run_dir, "weights_latest"), weights)

    # report epoch errors
    print()
    print(f"epoch mean prediction error   : {np.mean(epoch_mean_prediction_errors)}")
    print(f"epoch mean baseline error     : {np.mean(epoch_mean_baseline_errors)}")
    print(f"epoch mean extrapolation error: {np.mean(epoch_mean_extra_errors)}")

    fig = plot_error_curve(
        all_prediction_errors,
        all_baseline_errors,
        all_extra_errors,
        save_path=Path(run_dir, "error_curve.svg"),
        show=True
    )
    plt.close()

