import matplotlib.pyplot as plt
import numpy as np

def plot_state_prediction(p_s, p_z_pred, p_extra=None, delta_t=20, state_vars=[], save_path="", show=False):

    n_vars = p_s.shape[1]
    fig, axs = plt.subplots(n_vars, 1, sharex=True, sharey=True, figsize=(6, 4))
    plt.ylim(-1, 1)
    plt.suptitle(f"delta t = {delta_t}")
    legend = ["true state", "model prediction"]
    if p_extra is not None:
        legend = ["true state", "extrapolation", "model prediction"]
    for i in range(n_vars):
        timesteps = np.arange(0, p_s.shape[0])
        axs[i].plot(timesteps, p_s[:, i])
        if p_extra is not None:
            axs[i].plot(timesteps[2*delta_t:], p_extra[:, i])
        axs[i].plot(timesteps[delta_t:], p_z_pred[:-delta_t, i])
        axs[i].set_ylabel(state_vars[i], rotation=45)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)

    fig.legend(legend,
               loc="upper center",
               ncol=len(legend),
               bbox_to_anchor=(0.5, .95),
               frameon=False
               )
    plt.xlabel("t [steps]")
    plt.yticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

    return fig


def plot_error_curve(all_prediction_errors, all_baseline_errors, all_extra_errors,
                     t_delay="", save_path="", show=False):
    
    fig = plt.figure()
    plt.plot(range(len(all_prediction_errors)), all_prediction_errors)
    plt.hlines(np.mean(all_baseline_errors), 0, len(all_baseline_errors), colors="r")
    plt.hlines(np.mean(all_extra_errors), 0, len(all_extra_errors), colors="g")
    #plt.plot(range(len(all_baseline_errors)), all_baseline_errors)
    #plt.plot(range(len(all_extra_errors)), all_extra_errors)
    plt.xlabel("Example")
    plt.ylabel("Error")
    plt.legend(["current state", "linear extrapolation", "next state prediction"], frameon=False)
    plt.title(f"delay t = {t_delay}")
    plt.ylim(bottom=0)
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    
    return fig
