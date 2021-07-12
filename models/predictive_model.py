import nengo
import numpy as np
import scipy.linalg
from scipy.special import legendre

# TODO MAKE THE MODELS AS OBJECT CLASSES
# TODO ADD DOCUMENTATION

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


class LDN(nengo.Process):
    def __init__(self, theta, q, size_in=1):
        self.q = q  # number of internal state dimensions per input
        self.theta = theta  # size of time window (in seconds)
        self.size_in = size_in  # number of inputs

        # Do Aaron's math to generate the matrices A and B so that
        #  dx/dt = Ax + Bu will convert u into a legendre representation over a window theta
        #  https://github.com/arvoelke/nengolib/blob/master/nengolib/synapses/analog.py#L536
        A = np.zeros((q, q))
        B = np.zeros((q, 1))
        for i in range(q):
            B[i] = (-1.) ** i * (2 * i + 1)
            for j in range(q):
                A[i, j] = (2 * i + 1) * (-1 if i < j else (-1.) ** (i - j + 1))
        self.A = A / theta
        self.B = B / theta

        super().__init__(default_size_in=size_in, default_size_out=q * size_in)

    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        state = np.zeros((self.q, self.size_in))

        # Handle the fact that we're discretizing the time step
        #  https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
        Ad = scipy.linalg.expm(self.A * dt)
        Bd = np.dot(np.dot(np.linalg.inv(self.A), (Ad - np.eye(self.q))), self.B)

        # this code will be called every timestep
        def step_legendre(t, x, state=state):
            state[:] = np.dot(Ad, state) + np.dot(Bd, x[None, :])
            return state.T.flatten()

        return step_legendre

    def get_weights_for_delays(self, r):
        # compute the weights needed to extract the value at time r
        # from the network (r=0 is right now, r=1 is theta seconds ago)
        r = np.asarray(r)
        m = np.asarray([legendre(i)(2 * r - 1) for i in range(self.q)])
        return m.reshape(self.q, -1).T


class PredictiveModelLMU:
    def __init__(self, seed=42, neurons_per_dim=100, sample_freq=50,
                 lmu_theta=0.1, lmu_q=20, radius=1.5,
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
        self.radius = radius
        self.lmu_q = lmu_q
        self.lmu_theta = lmu_theta

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

            # make LMU unit
            ldn = nengo.Node(LDN(theta=self.lmu_theta, q=self.lmu_q, size_in=self.state_dim + self.action_dim))
            nengo.Connection(a, ldn[:self.action_dim])
            nengo.Connection(s, ldn[self.action_dim:])

            # make the hidden layer
            ens = nengo.Ensemble(
                n_neurons=self.neurons_per_dim * (self.state_dim + self.action_dim)*(1+self.lmu_q),
                dimensions=(self.state_dim + self.action_dim)*(1+self.lmu_q),
                neuron_type=nengo.LIFRate(),
                seed=self.seed
            )
            nengo.Connection(a, ens[:self.action_dim])
            nengo.Connection(s, ens[self.action_dim:self.action_dim+self.state_dim])
            nengo.Connection(ldn, ens[self.action_dim+self.state_dim:])

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

'''

LMU Nengo models and implementation

To use in network: ldn = nengo.Node(LDN(theta=theta, q=q))

'''





def make_model_LMU(action_df, state_df, weights=None, seed=42, n=100, samp_freq=50, lmu_theta=0.1, lmu_q=20,
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

        ldn = nengo.Node(LDN(theta=lmu_theta, q=lmu_q, size_in=5))

        nengo.Connection(a, ldn[0])
        nengo.Connection(s, ldn[1:])

            # make the hidden layer
        ens = nengo.Ensemble(n_neurons=n * 5, dimensions=5*lmu_q,
                             neuron_type=nengo.LIFRate(), seed=seed)

        #How do I connect each lmu to one dimension of ens?
        nengo.Connection(ldn, ens)


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


# The LMU2 model includes both the state and its temporal factorization in the ensemble to generate a prediction.

def make_model_LMU2(action_df, state_df, weights=None, seed=42, n=100, samp_freq=50, lmu_theta=0.1, lmu_q=20,
               t_delay=0.02, learning_rate=5e-5, radius=1.5):

    if weights is None:
        weights = np.zeros((5, n*6*(1+lmu_q)))

    model = nengo.Network()
    with model:
        model.config[nengo.Connection].synapse = None  # set the default synapse to None (normal default is 0.005s)

        # the input to the network
        def action_stim_func(t):
            return action_df["Q"].iloc[int(t * samp_freq)]

        a = nengo.Node(action_stim_func)

        # this function streams the state signal from file to node
        def state_stim_func(t):
            return  state_df["angle_sin"].iloc[int(t * samp_freq)], \
                    state_df["angle_cos"].iloc[int(t * samp_freq)], \
                    state_df["angleD"].iloc[int(t * samp_freq)], \
                    state_df["position"].iloc[int(t * samp_freq)], \
                    state_df["positionD"].iloc[int(t * samp_freq)]

        s = nengo.Node(state_stim_func)

        # the value to be predicted (which in this case is just the first dimension of the input)
        z = nengo.Node(None, size_in=5)
        nengo.Connection(s, z)

        z_pred = nengo.Node(None, size_in=5)

        ldn = nengo.Node(LDN(theta=lmu_theta, q=lmu_q, size_in=6))

        nengo.Connection(a, ldn[0])
        nengo.Connection(s, ldn[1:])

        # make the hidden layer
        ens = nengo.Ensemble(n_neurons=n*6*(1+lmu_q), dimensions=6*(1+lmu_q),
                             neuron_type=nengo.LIFRate(), seed=seed, radius=radius)

        #How do I connect each lmu to one dimension of ens?
        nengo.Connection(a, ens[:1])
        nengo.Connection(s, ens[1:6])
        nengo.Connection(ldn, ens[6:])


        # make the output weights we can learn
        conn = nengo.Connection(ens.neurons, z_pred,
                                transform=weights,  # change this if you have pre-recorded weights to use
                                learning_rule_type=nengo.PES(learning_rate=learning_rate,
                                                             pre_synapse=DiscreteDelay(t_delay)
                                                             # delay the activity value when updating weights
                                                             ))

        # compute the error by subtracting the current measurement from a delayed version of the predicton
        error = nengo.Node(None, size_in=5)
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

def make_model_LMU3(action_df, state_df, weights=None, seed=42, n=100, samp_freq=50, lmu_theta=0.1, lmu_q=20,   # LMU2 but with 'angle' state
                        t_delay=0.02, learning_rate=5e-5, radius=1.5):

    if weights is None:
        weights = np.zeros((6, n * 7 * (1 + lmu_q)))

    model = nengo.Network()
    with model:
        model.config[nengo.Connection].synapse = None  # set the default synapse to None (normal default is 0.005s)

        # the input to the network
        def action_stim_func(t):
            return action_df["Q"].iloc[int(t * samp_freq)]

        a = nengo.Node(action_stim_func)

        # this function streams the state signal from file to node
        def state_stim_func(t):
            return state_df["angle"].iloc[int(t * samp_freq)], \
                    state_df["angleD"].iloc[int(t * samp_freq)], \
                    state_df["angle_sin"].iloc[int(t * samp_freq)], \
                    state_df["angle_cos"].iloc[int(t * samp_freq)], \
                    state_df["position"].iloc[int(t * samp_freq)], \
                    state_df["positionD"].iloc[int(t * samp_freq)]

        s = nengo.Node(state_stim_func)

        # the value to be predicted (which in this case is just the first dimension of the input)
        z = nengo.Node(None, size_in=6)
        nengo.Connection(s, z)

        z_pred = nengo.Node(None, size_in=6)

        ldn = nengo.Node(LDN(theta=lmu_theta, q=lmu_q, size_in=7))

        nengo.Connection(a, ldn[0])
        nengo.Connection(s, ldn[1:])

        # make the hidden layer
        ens = nengo.Ensemble(n_neurons=n * 7 * (1 + lmu_q), dimensions=7 * (1 + lmu_q),
                                 neuron_type=nengo.LIFRate(), seed=seed, radius=radius)

        # How do I connect each lmu to one dimension of ens?
        nengo.Connection(a, ens[:1])
        nengo.Connection(s, ens[1:7])
        nengo.Connection(ldn, ens[7:])

        # make the output weights we can learn
        conn = nengo.Connection(ens.neurons, z_pred,
                                transform=weights,  # change this if you have pre-recorded weights to use
                                learning_rule_type=nengo.PES(learning_rate=learning_rate,
                                                            pre_synapse=DiscreteDelay(t_delay)
                                                            # delay the activity value when updating weights
                                                            ))

        # compute the error by subtracting the current measurement from a delayed version of the predicton
        error = nengo.Node(None, size_in=6)
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

# class PredictiveModel_LMU:
#     def __init__(self, seed=42, neurons_per_dim=100, sample_freq=50, lmu_theta=1, lmu_q=10,
#                  t_delay=0.02, learning_rate=5e-5, action_vars=["Q"],
#                  state_vars=["angle_sin", "angleD", "position", "positionD"]):
#         self.seed = seed
#         self.neurons_per_dim = neurons_per_dim
#         self.sample_freq = sample_freq
#         self.lmu_q=lmu_q
#         self.lmu_theta=lmu_theta 
#         self.t_delay = t_delay
#         self.learning_rate = learning_rate
#         self.action_vars = action_vars
#         self.action_dim = len(action_vars)
#         self.state_vars = state_vars
#         self.state_dim = len(state_vars)

#     def make_model(self, action_df, state_df, weights=None):

#         for v in self.action_vars:
#             assert v in action_df.columns
#         for v in self.state_vars:
#             assert v in state_df.columns

#         if weights is None:
#             weights = np.zeros((
#                 self.state_dim,
#                 self.neurons_per_dim * (self.state_dim + self.action_dim))
#             )

#         model = nengo.Network()
#         with model:
#             # set the default synapse to None (normal default is 0.005s)
#             model.config[nengo.Connection].synapse = None

#             # the input to the network
#             def action_stim_func(t):
#                 # I have no idea if this works...
#                 r = [action_df[x].iloc[int(t * self.sample_freq)] for x in self.action_vars]
#                 return tuple(r)

#             a = nengo.Node(action_stim_func)

#             # this function streams the state signal from file to node
#             def state_stim_func(t):
#                 r = [state_df[x].iloc[int(t * self.sample_freq)] for x in self.state_vars]
#                 return tuple(r)

#             s = nengo.Node(state_stim_func)

#             # the value to be predicted (which in this case is just the first dimension of the input)
#             z = nengo.Node(None, size_in=self.state_dim)
#             nengo.Connection(s, z)

#             z_pred = nengo.Node(None, size_in=self.state_dim)


#             # make the LDN layer

#             ldn1 = nengo.Node(LDN(theta=self.lmu_theta, q=self.lmu_q, size_in=self.state_dim + self.action_dim))
#             nengo.Connection(a, ldn1[:self.action_dim])
#             nengo.Connection(s, ldn1[self.action_dim:])

#             # make the hidden layer
#             ens = nengo.Ensemble(
#                 n_neurons=self.neurons_per_dim * (self.state_dim + self.action_dim),
#                 dimensions=(self.state_dim + self.action_dim),
#                 neuron_type=nengo.LIFRate(),
#                 seed=self.seed
#             )

#             nengo.Connection(ldn1, ens)

#             # nengo.Connection(a, ens[:self.action_dim])
#             # nengo.Connection(s, ens[self.action_dim:])

#             # make the output weights we can learn
#             conn = nengo.Connection(ens.neurons, z_pred,
#                                     transform=weights,  # change this if you have pre-recorded weights to use
#                                     learning_rule_type=nengo.PES(learning_rate=self.learning_rate,
#                                                                  pre_synapse=DiscreteDelay(self.t_delay)
#                                                                  # delay the activity value when updating weights
#                                                                  ))

#             # compute the error by subtracting the current measurement from a delayed version of the predicton
#             error = nengo.Node(None, size_in=self.state_dim)
#             nengo.Connection(z_pred, error, synapse=DiscreteDelay(self.t_delay))
#             nengo.Connection(z, error, transform=-1)
#             # apply the error to the learning rule
#             nengo.Connection(error, conn.learning_rule)

#             # record the input to the network
#             p_a = nengo.Probe(a)
#             p_s = nengo.Probe(s)
#             # record the value to be predicted
#             p_z = nengo.Probe(z)
#             # record the prediction
#             p_z_pred = nengo.Probe(z_pred)
#             # record the error
#             p_e = nengo.Probe(error)
#             # record the weights (but only every 0.1 seconds just to save memory)
#             p_weights = nengo.Probe(conn, 'weights', sample_every=0.1)

#         return model, [p_a, p_s, p_z, p_z_pred, p_e, p_weights]


