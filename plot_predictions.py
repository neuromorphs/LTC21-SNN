import matplotlib.pyplot as plt
import numpy as np

def plot_state_prediction(p_s, p_z_pred, p_extra=None, delta_t=20, save_path="", show=False):

    n_vars = p_s.shape[1]
    fig, axs = plt.subplots(n_vars, 1, sharex=True, figsize=(6, 4))
    for i in range(n_vars):
        timesteps = np.arange(0, p_s.shape[0])
        axs[i].plot(timesteps, p_s[:, i])
        axs[i].plot(timesteps[delta_t:], p_z_pred[:-delta_t, i])
        if p_extra is not None:
            axs[i].plot(timesteps[2*delta_t:], p_extra[:, i])
    legend = ["true state", "model prediction"]
    if p_extra is not None:
        legend += ["extrapolation"]
    plt.legend(legend,
               loc="upper center",
               ncol=len(legend),
               bbox_to_anchor=(0.5, 5.2)
               )
    plt.xlabel("t")

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

    return fig


def plot_error_curve(all_prediction_errors, all_baseline_errors, all_extra_errors, save_path="", show=False):
    
    fig = plt.figure()
    plt.plot(range(len(all_prediction_errors)), all_prediction_errors)
    plt.plot(range(len(all_baseline_errors)), all_baseline_errors)
    plt.plot(range(len(all_extra_errors)), all_extra_errors)
    plt.xlabel("Example")
    plt.ylabel("Error")
    plt.legend(["next state prediction", "current state", "linear extrapolation"])
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    
    return fig
