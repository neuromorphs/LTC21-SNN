import numpy as np
import nengo

# move this to a model file
class DiscreteDelay(nengo.synapses.Synapse):
    def __init__(self, delay, size_in=1):
        self.delay = delay
        super().__init__(default_size_in=size_in, default_size_out=size_in)

    def make_state(self, shape_in, shape_out, dt, dtype=None, y0=None):
        return {}

    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        steps = int(self.delay/dt)
        if steps == 0:
            def step_delay(t, x):
                return x
            return step_delay
        assert steps > 0

        state = np.zeros((steps, shape_in[0]))
        state_index = np.array([0])

        def step_delay(t, x, state=state, state_index=state_index):
            result = state[state_index]
            state[state_index] = x
            state_index[:] = (state_index + 1) % state.shape[0]
            return result

        return step_delay

class PredictiveModel:
    def __init__(self, seed=42, neurons_per_dim=100, sample_freq=50,
                 t_delay=0.02, learning_rate=5e-5, action_vars=["Q"],
                 state_vars=["angle_sin", "angleD", "position", "positionD"]):
        self.seed = seed
        self.neurons_per_dim = neurons_per_dim
        self.sample_freq = sample_freq
        self.t_delay = t_delay
        self.learning_rate = learning_rate
        self.action_vars = action_vars
        self.action_dim = len(action_vars)
        self.state_vars = state_vars
        self.state_dim = len(state_vars)

    def make_model(self, action_df, state_df, weights=None):

        for v in self.action_vars:
            assert v in action_df.columns
        for v in self.state_vars:
            assert v in state_df.columns

        if weights is None:
            weights = np.zeros((
                self.state_dim,
                self.neurons_per_dim * (self.state_dim + self.action_dim))
            )

        model = nengo.Network()
        with model:
            # set the default synapse to None (normal default is 0.005s)
            model.config[nengo.Connection].synapse = None

            # the input to the network
            def action_stim_func(t):
                # I have no idea if this works...
                r = [action_df[x].iloc[int(t * self.sample_freq)] for x in self.action_vars]
                return tuple(r)

            a = nengo.Node(action_stim_func)

            # this function streams the state signal from file to node
            def state_stim_func(t):
                r = [state_df[x].iloc[int(t * self.sample_freq)] for x in self.state_vars]
                return tuple(r)

            s = nengo.Node(state_stim_func)

            # the value to be predicted (which in this case is just the first dimension of the input)
            z = nengo.Node(None, size_in=self.state_dim)
            nengo.Connection(s, z)

            z_pred = nengo.Node(None, size_in=self.state_dim)

            # make the hidden layer
            ens = nengo.Ensemble(
                n_neurons=self.neurons_per_dim * (self.state_dim + self.action_dim),
                dimensions=(self.state_dim + self.action_dim),
                neuron_type=nengo.LIFRate(),
                seed=self.seed
            )
            nengo.Connection(a, ens[:self.action_dim])
            nengo.Connection(s, ens[self.action_dim:])

            # make the output weights we can learn
            conn = nengo.Connection(ens.neurons, z_pred,
                                    transform=weights,  # change this if you have pre-recorded weights to use
                                    learning_rule_type=nengo.PES(learning_rate=self.learning_rate,
                                                                 pre_synapse=DiscreteDelay(self.t_delay)
                                                                 # delay the activity value when updating weights
                                                                 ))

            # compute the error by subtracting the current measurement from a delayed version of the predicton
            error = nengo.Node(None, size_in=self.state_dim)
            nengo.Connection(z_pred, error, synapse=DiscreteDelay(self.t_delay))
            nengo.Connection(z, error, transform=-1)
            # apply the error to the learning rule
            nengo.Connection(error, conn.learning_rule)

            # record the input to the network
            p_a = nengo.Probe(a)
            p_s = nengo.Probe(s)
            # record the value to be predicted
            p_z = nengo.Probe(z)
            # record the prediction
            p_z_pred = nengo.Probe(z_pred)
            # record the error
            p_e = nengo.Probe(error)
            # record the weights (but only every 0.1 seconds just to save memory)
            p_weights = nengo.Probe(conn, 'weights', sample_every=0.1)

        return model, [p_a, p_s, p_z, p_z_pred, p_e, p_weights]


# move this to a model file
def make_model(action_df, state_df, weights=None, seed=42, n=100, samp_freq=50,
               t_delay=0.02, learning_rate=5e-5):
    if weights is None:
        weights = np.zeros((4, n * 5))

    model = nengo.Network()
    with model:
        model.config[nengo.Connection].synapse = None  # set the default synapse to None (normal default is 0.005s)

        # the input to the network
        def action_stim_func(t):
            return action_df["Q"].iloc[int(t * samp_freq)]

        a = nengo.Node(action_stim_func)

        # this function streams the state signal from file to node
        def state_stim_func(t):
            return state_df["angle_sin"].iloc[int(t * samp_freq)], \
                   state_df["angleD"].iloc[int(t * samp_freq)], \
                   state_df["position"].iloc[int(t * samp_freq)], \
                   state_df["positionD"].iloc[int(t * samp_freq)]

        s = nengo.Node(state_stim_func)

        # the value to be predicted (which in this case is just the first dimension of the input)
        z = nengo.Node(None, size_in=4)
        nengo.Connection(s, z)

        z_pred = nengo.Node(None, size_in=4)

        # make the hidden layer
        ens = nengo.Ensemble(n_neurons=n * 5, dimensions=5,
                             neuron_type=nengo.LIFRate(), seed=seed)
        nengo.Connection(a, ens[0])
        nengo.Connection(s, ens[1:])

        # make the output weights we can learn
        conn = nengo.Connection(ens.neurons, z_pred,
                                transform=weights,  # change this if you have pre-recorded weights to use
                                learning_rule_type=nengo.PES(learning_rate=learning_rate,
                                                             pre_synapse=DiscreteDelay(t_delay)
                                                             # delay the activity value when updating weights
                                                             ))

        # compute the error by subtracting the current measurement from a delayed version of the predicton
        error = nengo.Node(None, size_in=4)
        nengo.Connection(z_pred, error, synapse=DiscreteDelay(t_delay))
        nengo.Connection(z, error, transform=-1)
        # apply the error to the learning rule
        nengo.Connection(error, conn.learning_rule)

        # record the input to the network
        p_a = nengo.Probe(a)
        p_s = nengo.Probe(s)
        # record the value to be predicted
        p_z = nengo.Probe(z)
        # record the prediction
        p_z_pred = nengo.Probe(z_pred)
        # record the error
        p_e = nengo.Probe(error)
        # record the weights (but only every 0.1 seconds just to save memory)
        p_weights = nengo.Probe(conn, 'weights', sample_every=0.1)

    return model, [p_a, p_s, p_z, p_z_pred, p_e, p_weights]