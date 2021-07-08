import pandas as pd
import numpy as np
import nengo
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("Qt5Agg")
import os
from tqdm import tqdm
from pathlib import Path
import datetime
from models.predictive_model import make_model, make_model_LMU, make_model_LMU2
from plot_predictions import plot_state_prediction

model_name = "LMU2"
experiment_name = "test1"
data_dir = "data/Validate/"
results_dir = "results/"
load_weights = "results/test1/LMU2/2021-07-08_11.24.25.863527/weights_latest.npy"

assert Path(load_weights).is_file()

bound = 0.19  # if the cart ever leaves these bounds, the data is ignored

# TODO These parameters should be loaded from some model state_dict!
samp_freq = 50  # cartpole data is recorded at ~50Hz
dt = 0.001  # nengo time step
learning_rate = 0  # lr should be 0 when testing
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
test_data = []
for _, _, files in os.walk(data_dir):
    for f in files:
        df = pd.read_csv(data_dir + f, skiprows=28)
        # Filter datasets that might have bounced of the edges of the track
        if df["position"].min() < -bound or df["position"].max() > bound:
            continue
        test_data.append(df)
    print(f"test data contains {len(test_data)} files")

# init weights from file or empty
print("loading weights from", load_weights)
weights = np.load(load_weights)

# test the model
all_prediction_errors = []
all_baseline_errors = []
all_extra_errors = []
print("\nstarting model evaluation")
with tqdm(total=len(test_data)) as t:
    for i, df in enumerate(test_data):
        action_df = df[["time", "Q"]]
        state_df = df[
            [
                "time",
                # "angle",
                "angleD",
                # "angleDD",
                # "angle_cos",
                "angle_sin",
                "position",
                "positionD",
                # "positionDD",
                # "target_position",
            ]
        ]
        t_max = action_df["time"].max()  # number of seconds to run
        # TODO use the OO method of creating the model (see predictive_model.py)

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
        all_prediction_errors.append(mean_prediction_error)

        # report the difference between prediction and last state
        mean_baseline_error = np.mean(np.abs(p_z_pred - p_s))
        all_baseline_errors.append(mean_baseline_error)

        # report the difference between prediction and linear extrapolation
        p_s_extrapolation = 2 * p_s[50:] - p_s[:-50]
        mean_extrapolation_error = np.mean(np.abs(p_s_extrapolation - p_z[50:]))
        all_extra_errors.append(mean_extrapolation_error)

        # update the loading bar
        t.set_postfix(loss="{:05.4f}".format(mean_prediction_error))
        t.update()

        plot_state_prediction(p_z, p_z_pred)

        plt.figure()
        plt.plot(range(len(p_z)), p_z)
        plt.plot(range(len(p_z_pred)), p_z_pred)
        plt.plot(range(len(p_s_extrapolation)), p_s_extrapolation)
        plt.legend(["p_z A", "p_z B", "p_z C", "p_z D",
                    "pred A", "pred B", "pred C", "pred D",
                    "extra A", "extra B", "extra C", "extra D"])
        plt.show()
        plt.close()

        break

# report epoch errors
print()
print(f"mean prediction error   : {np.mean(all_prediction_errors)}")
print(f"mean baseline error     : {np.mean(all_baseline_errors)}")
print(f"mean extrapolation error: {np.mean(all_extra_errors)}")

plt.plot(range(len(all_prediction_errors)), all_prediction_errors)
plt.plot(range(len(all_baseline_errors)), all_baseline_errors)
plt.plot(range(len(all_extra_errors)), all_extra_errors)
plt.xlabel("Example")
plt.ylabel("Error")
plt.legend(["next state prediction", "current state", "linear extrapolation"])
plt.savefig(Path(run_dir, "error_curve.svg"))
plt.show()
plt.close()
