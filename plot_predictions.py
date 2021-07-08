import matplotlib.pyplot as plt

def plot_state_prediction(p_s, p_z_pred):

    n_vars = p_s.shape[1]
    fig, axs = plt.subplots(n_vars, 1, sharex=True, figsize=(6, 4))
    for i in range(n_vars):
        axs[i].plot(p_s[:, i])
        axs[i].plot(p_z_pred[:, i])



    plt.show()

    return fig