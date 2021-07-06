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
from models.predictive_model import make_model
from models.predictive_model import make_model_LMU

LMU_enabled = True

experiment_name = "test1"
data_dir = "data/Validate/"
results_dir = "results/"
load_weights = "results/test1/2021-07-06_17.48.05.498405/weights_latest.npy"

assert Path(load_weights).is_file()

bound = 0.19            # if the cart ever leaves these bounds, the data is ignored

# TODO These parameters should be loaded from some model state_dict!
samp_freq = 50          # cartpole data is recorded at ~50Hz
dt = 0.001              # nengo time step
learning_rate = 0       # lr
t_delay = 0.02          # how far to predict the future (initial guess)
neurons_per_dim = 100   # number of neurons representing each dimension
seed = 4                # to get reproducible neuron properties across runs

# crating a unique folder to save the weights in
folder_name = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
run_dir = Path(results_dir, experiment_name, folder_name)
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
all_errors = []
all_baseline_errors = []
print("\nstarting model evaluation")
epoch_mean_errors = []
epoch_baseline_errors = []
with tqdm(total=len(test_data)) as t:
    for i, df in enumerate(test_data):
        action_df = df[["time", "Q"]]
        state_df = df[["time",
                       # "angle",
                       "angleD",
                       # "angleDD",
                       # "angle_cos",
                       "angle_sin",
                       "position",
                       "positionD",
                       # "positionDD",
                       # "target_position",
                       ]]
        t_max = action_df["time"].max()   # number of seconds to run
        # TODO use the OO method of creating the model (see predictive_model.py)

        if LMU_enabled:

          model, recordings = make_model_LMU(
              action_df,
              state_df,
              weights=weights,
              seed=seed,
              n=neurons_per_dim,
              samp_freq=samp_freq,
              t_delay=t_delay,
              learning_rate=learning_rate
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
              learning_rate=learning_rate
          )

        sim = nengo.Simulator(model, progress_bar=False)
        sim.run(t_max)
        #weights = sim.data[recordings[P_WEIGHTS]][-1]
        p_e = sim.data[recordings[P_E]]
        p_z = sim.data[recordings[P_Z]]
        p_z_pred = sim.data[recordings[P_Z_PRED]]
        p_s = sim.data[recordings[P_S]]

        mean_error = np.mean(np.abs(p_e))
        baseline_error = np.mean(np.abs(p_z_pred - p_s))
        epoch_mean_errors.append(mean_error)
        epoch_baseline_errors.append(baseline_error)
        all_errors.append(mean_error)
        all_baseline_errors.append(baseline_error)
        t.set_postfix(loss="{:05.4f}".format(mean_error))
        t.update()

print(f"\nepoch mean loss: {np.mean(epoch_mean_errors)}")

plt.plot(range(len(all_errors)), all_errors)
plt.plot(range(len(all_baseline_errors)), all_baseline_errors)
plt.xlabel("Example")
plt.ylabel("Error")
plt.legend(["Prediction", "Baseline"])
plt.savefig(Path(run_dir, "error_curve_test.svg"))
# plt.show()
plt.close()