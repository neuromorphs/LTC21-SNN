import nengo
import numpy as np
import pandas as pd
import scipy.linalg
from scipy.special import legendre
import pickle

# TODO ADD DOCUMENTATION

# This code is completely taken from Terry Steward:
# We'll make a simple object to implement the delayed connection
class Delay:
    def __init__(self, dimensions, timesteps=50):
        self.history = np.zeros((timesteps, dimensions))

    def step(self, t, x):
        self.history = np.roll(self.history, -1)
        self.history[-1] = x
        return self.history[0]

# This code is completely taken from Terry Steward:
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

# This code is completely taken from Terry Steward:
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


class DataParser:
    def __init__(self, data_df, sample_freq, vars=[]):
        self.data_df = data_df
        self.sample_freq = sample_freq
        self.vars = vars

    def parse_data(self, t):
        r = [self.data_df[x].iloc[int(t * self.sample_freq)] for x in self.vars]
        return r

    def update_data(self, data_df):
        self.data_df = data_df


class PredictiveModelLMU:
    def __init__(self, seed=42, neurons_per_dim=100, sample_freq=50,
                 lmu_theta=0.1, lmu_q=20, radius=1.5, dt=0.001,
                 t_delays=[0.02], learning_rate=5e-5, action_vars=["Q"],
                 state_vars=["angle_sin", "angle_cos", "angleD", "position", "positionD"],
                 action_df=None, state_df=None, weights=None, scales={},
                 predict_delta=True, *args, **kwargs):

        self.seed = seed
        self.neurons_per_dim = neurons_per_dim
        self.sample_freq = sample_freq
        self.t_delays = t_delays
        self.learning_rate = learning_rate
        self.action_vars = action_vars
        self.action_dim = len(action_vars)
        self.state_vars = state_vars
        self.state_dim = len(state_vars)
        self.radius = radius
        self.dt = dt
        self.lmu_q = lmu_q
        self.lmu_theta = lmu_theta
        self.weights = weights
        self.scales = scales
        self.predict_delta = predict_delta

        if action_df is None:
            self.action_df = pd.DataFrame(
                np.zeros((1, len(action_vars) + 1)),
                columns=["time"] + action_vars,
            )

        if state_df is None:
            self.state_df = pd.DataFrame(
                np.zeros((1, len(state_vars) + 1)),
                columns=["time"] + state_vars,
            )

        self.action_parser = DataParser(
            data_df=self.action_df,
            sample_freq=self.sample_freq,
            vars=self.action_vars
        )

        self.state_parser = DataParser(
            data_df=self.state_df,
            sample_freq=self.sample_freq,
            vars=self.state_vars
        )

        # this function streams the state signal from file to node
        def action_stim_func(t):
            return self.action_parser.parse_data(t)

        # this function streams the state signal from file to node
        def state_stim_func(t):
            return self.state_parser.parse_data(t)

        self.action_stim_func = action_stim_func
        self.state_stim_func = state_stim_func

        self.model, self.recordings = self.make_model()
        self.sim = nengo.Simulator(self.model, dt=self.dt, progress_bar=False)

    def set_inputs(self, action_df, state_df):

        for v in self.action_vars + ["time"]:
            assert v in action_df.columns
        for v in self.state_vars:
            assert v in state_df.columns

        self.action_df = action_df
        self.state_df = state_df

        self.action_parser.update_data(self.action_df)
        self.state_parser.update_data(self.state_df)

    def reset_sim(self):

        self.sim.reset(seed=self.seed)

    def set_weights(self, weights):

        weights = np.array(weights)
        assert weights.shape == (len(self.t_delays),
                                 self.state_dim,
                                 self.neurons_per_dim * (self.state_dim + self.action_dim) * (1 + self.lmu_q))

        self.weights = weights
        for i, con in enumerate(self.connections):
            self.sim.signals[self.sim.model.sig[con]["weights"]] = self.weights[i]

    def get_weights(self):

        weights = []
        for con in self.connections:
            weights.append(self.sim.signals[self.sim.model.sig[con]["weights"]])
        return np.array(weights)

    def process_files(self):

        t_max = self.action_df["time"].max()  # number of seconds to run
        self.sim.run(t_max)

        return self.recordings

    def get_state_dict(self):

        state_dict = {
            "seed": self.seed,
            "neurons_per_dim": self.neurons_per_dim,
            "sample_freq": self.sample_freq,
            "t_delays": self.t_delays,
            "learning_rate": self.learning_rate,
            "action_vars": self.action_vars,
            "state_vars": self.state_vars,
            "radius": self.radius,
            "dt": self.dt,
            "lmu_q": self.lmu_q,
            "lmu_theta": self.lmu_theta,
            "weights": self.get_weights(),
            "scales": self.scales,
            "predict_delta": self.predict_delta
        }

        return state_dict

    def save_state_dict(self, path="model_state.pkl"):

        state_dict = self.get_state_dict()
        with open(path, "wb") as f:
            pickle.dump(state_dict, f)

    def make_model(self):

        if self.weights is None:
            self.weights = np.zeros((
                len(self.t_delays),
                self.state_dim,
                self.neurons_per_dim * (self.state_dim + self.action_dim) * (1 + self.lmu_q))
            )

        model = nengo.Network()
        with model:
            # set the default synapse to None (normal default is 0.005s)
            model.config[nengo.Connection].synapse = None

            # initialize input nodes
            a = nengo.Node(self.action_stim_func)
            s = nengo.Node(self.state_stim_func)

            # record the input to the network
            p_a = nengo.Probe(a)
            p_s = nengo.Probe(s)

            # the value to be predicted (which in this case is just the first dimension of the input)
            z = nengo.Node(None, size_in=self.state_dim)
            nengo.Connection(s, z)

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

            z_preds = []
            self.connections = []
            errors = []

            recordings = {
                "states" : p_s,
                "actions": p_a,
                "predictions": {},
            }

            for i, t_d in enumerate(self.t_delays):

                z_preds.append(nengo.Node(None, size_in=self.state_dim))

                # make the output weights we can learn
                self.connections.append(
                    nengo.Connection(
                        ens.neurons,
                        z_preds[-1],
                        transform=self.weights[i],  # change this if you have pre-recorded weights to use
                        seed=self.seed,
                        learning_rule_type=nengo.PES(
                            learning_rate=self.learning_rate,
                            pre_synapse=DiscreteDelay(t_d)  # delay the activity value when updating weights
                        )
                    )
                )

                # compute the error by subtracting the current measurement from a delayed version of the predicton
                errors.append(nengo.Node(None, size_in=self.state_dim))
                nengo.Connection(z_preds[-1], errors[-1], synapse=DiscreteDelay(t_d))
                nengo.Connection(s, errors[-1], transform=-1)
                # if wanted, have the model predict the difference from the last state instead
                if self.predict_delta:
                    nengo.Connection(z, z_preds[-1])

                # apply the error to the learning rule
                nengo.Connection(errors[-1], self.connections[-1].learning_rule)

                prediction = {
                    f"{i}": {
                        "delay": t_d,
                        # record the prediction
                        "states_pred": nengo.Probe(z_preds[-1]),
                        # record the error
                        "errors": nengo.Probe(errors[-1]),
                    }
                }

                recordings["predictions"].update(prediction)

        return model, recordings



class PredictiveModelLMU_test:
    def __init__(self, seed=42, neurons_per_dim=100, sample_freq=50,
                 lmu_theta=0.1, lmu_q=20, radius=1.5, dt=0.001,
                 t_delays=[0.02], learning_rate=5e-5, action_vars=["Q"],
                 state_vars=["angle_sin", "angle_cos", "angleD", "position", "positionD"],
                 action_df=None, state_df=None, weights=None, scales={},
                 predict_delta=True, *args, **kwargs):

        self.seed = seed
        self.neurons_per_dim = neurons_per_dim
        self.sample_freq = sample_freq
        self.t_delays = t_delays
        self.learning_rate = learning_rate
        self.action_vars = action_vars
        self.action_dim = len(action_vars)
        self.state_vars = state_vars
        self.state_dim = len(state_vars)
        self.radius = radius
        self.dt = dt
        self.lmu_q = lmu_q
        self.lmu_theta = lmu_theta
        self.weights = weights
        self.scales = scales
        self.predict_delta = predict_delta

        if action_df is None:
            self.action_df = pd.DataFrame(
                np.zeros((1, len(action_vars) + 1)),
                columns=["time"] + action_vars,
            )

        if state_df is None:
            self.state_df = pd.DataFrame(
                np.zeros((1, len(state_vars) + 1)),
                columns=["time"] + state_vars,
            )

        self.action_parser = DataParser(
            data_df=self.action_df,
            sample_freq=self.sample_freq,
            vars=self.action_vars
        )

        self.state_parser = DataParser(
            data_df=self.state_df,
            sample_freq=self.sample_freq,
            vars=self.state_vars
        )

        # this function streams the state signal from file to node
        def action_stim_func(t):
            return self.action_parser.parse_data(t)

        # this function streams the state signal from file to node
        def state_stim_func(t):
            return self.state_parser.parse_data(t)

        self.action_stim_func = action_stim_func
        self.state_stim_func = state_stim_func

        self.model, self.recordings = self.make_model()
        self.sim = nengo.Simulator(self.model, dt=self.dt, progress_bar=False)

    def set_inputs(self, action_df, state_df):

        for v in self.action_vars + ["time"]:
            assert v in action_df.columns
        for v in self.state_vars:
            assert v in state_df.columns

        self.action_df = action_df
        self.state_df = state_df

        self.action_parser.update_data(self.action_df)
        self.state_parser.update_data(self.state_df)

    def reset_sim(self):

        self.sim.reset(seed=self.seed)

    def set_weights(self, weights):

        weights = np.array(weights)
        assert weights.shape == (len(self.t_delays),
                                 self.state_dim,
                                 self.neurons_per_dim * (self.state_dim + self.action_dim) * (1 + self.lmu_q))

        self.weights = weights
        for i, con in enumerate(self.connections):
            self.sim.signals[self.sim.model.sig[con]["weights"]] = self.weights[i]

    def get_weights(self):

        weights = []
        for con in self.connections:
            weights.append(self.sim.signals[self.sim.model.sig[con]["weights"]])
        return np.array(weights)

    def process_files(self):

        t_max = self.action_df["time"].max()  # number of seconds to run
        self.sim.run(t_max)

        return self.recordings

    def get_state_dict(self):

        state_dict = {
            "seed": self.seed,
            "neurons_per_dim": self.neurons_per_dim,
            "sample_freq": self.sample_freq,
            "t_delays": self.t_delays,
            "learning_rate": self.learning_rate,
            "action_vars": self.action_vars,
            "state_vars": self.state_vars,
            "radius": self.radius,
            "dt": self.dt,
            "lmu_q": self.lmu_q,
            "lmu_theta": self.lmu_theta,
            "weights": self.get_weights(),
            "scales": self.scales,
            "predict_delta": self.predict_delta
        }

        return state_dict

    def save_state_dict(self, path="model_state.pkl"):

        state_dict = self.get_state_dict()
        with open(path, "wb") as f:
            pickle.dump(state_dict, f)

    def make_model(self):

        if self.weights is None:
            self.weights = np.zeros((
                len(self.t_delays),
                self.state_dim,
                self.neurons_per_dim * (self.state_dim + self.action_dim) * (1 + self.lmu_q))
            )

        model = nengo.Network()
        with model:
            # set the default synapse to None (normal default is 0.005s)
            model.config[nengo.Connection].synapse = None

            # initialize input nodes
            a = nengo.Node(self.action_stim_func)
            s = nengo.Node(self.state_stim_func)

            # record the input to the network
            p_a = nengo.Probe(a)
            p_s = nengo.Probe(s)

            recordings = {
                "states" : p_s,
                "actions": p_a,
                "predictions": {},
                #"grad" : nengo.Probe(grad)
            }

            # make some lists to store some objects
            z_preds = []
            self.connections = []
            errors = []
            ensembles = []
            ldns = []

            ###### THIS IS SUPER EXPERIMENTAL:
            """
            # create a memory node for the last three states
            m = nengo.Node(None, size_in=self.state_dim * 3)
            nengo.Connection(s, m[:self.state_dim])
            nengo.Connection(m[:self.state_dim], m[self.state_dim:2*self.state_dim], synapse=DiscreteDelay(0.02))
            nengo.Connection(m[self.state_dim:2*self.state_dim], m[2*self.state_dim:], synapse=DiscreteDelay(0.019))

            # create a node that calculates the approximate gradient of the state variables based on the memory
            def appoximate_grad(t, x):
                x1 = x[:self.state_dim]
                x2 = x[self.state_dim:2*self.state_dim]
                x3 = x[2*self.state_dim:]
                f_x = (1 * x3 - 4 * x2 + 3 * x1) / 0.04
                return f_x

            def appoximate_grad2(t, x):
                if t <= 0.04 * 2:
                    return np.zeros(self.state_dim)
                x1 = x[:self.state_dim]
                x2 = x[self.state_dim:2*self.state_dim]
                #x3 = x[2*self.state_dim:]
                f_x = (x1 - x2) / 0.02
                return f_x

            grad = nengo.Node(appoximate_grad2, size_in=self.state_dim*3, size_out=self.state_dim)
            nengo.Connection(m, grad, synapse=0.0)
            """
            ######

            for i, t_d in enumerate(self.t_delays):

                # make a node that represents the current state prediction
                z_preds.append(nengo.Node(None, size_in=self.state_dim))

                # make LMU unit
                ldns.append(nengo.Node(LDN(
                    theta=self.lmu_theta,
                    q=self.lmu_q,
                    size_in=self.state_dim + self.action_dim)
                ))
                nengo.Connection(a, ldns[-1][:self.action_dim])

                # make the hidden layer
                ensembles.append(nengo.Ensemble(
                    n_neurons=self.neurons_per_dim * (self.state_dim + self.action_dim)*(1+self.lmu_q),
                    dimensions=(self.state_dim + self.action_dim)*(1+self.lmu_q),
                    neuron_type=nengo.LIFRate(),
                    seed=self.seed
                ))
                nengo.Connection(a, ensembles[-1][:self.action_dim])
                nengo.Connection(ldns[-1], ensembles[-1][self.action_dim+self.state_dim:])

                # make the output weights we can learn
                self.connections.append(
                    nengo.Connection(
                        ensembles[-1].neurons,
                        z_preds[-1],
                        transform=self.weights[i],  # change this if you have pre-recorded weights to use
                        seed=self.seed,
                        learning_rule_type=nengo.PES(
                            learning_rate=self.learning_rate,
                            pre_synapse=DiscreteDelay(t_d)  # delay the activity value when updating weights
                        )
                    )
                )

                # compute the error by subtracting the current measurement from a delayed version of the predicton
                errors.append(nengo.Node(None, size_in=self.state_dim))
                nengo.Connection(z_preds[-1], errors[-1], synapse=DiscreteDelay(t_d))
                nengo.Connection(s, errors[-1], transform=-1)

                delay = Delay(self.state_dim, timesteps=int((t_d - self.dt) / self.dt))

                def switch_input(t, x):
                    if t < 1.0:
                        return x[:self.state_dim]
                    return x[self.state_dim:]

                switch_node = nengo.Node(switch_input, size_in=self.state_dim*2, size_out=self.state_dim)
                delay_node = nengo.Node(delay.step, size_in=self.state_dim)

                # if wanted, have the model predict the difference from the last state instead
                if self.predict_delta:
                    #nengo.Connection(z, z_preds[-1], transform=1)
                    nengo.Connection(z_preds[-1], delay_node, synapse=0.01) #, synapse=DiscreteDelay(t_d))
                    nengo.Connection(delay_node, switch_node[self.state_dim:])
                    nengo.Connection(s, switch_node[:self.state_dim])
                    nengo.Connection(switch_node, z_preds[-1])
                    nengo.Connection(switch_node, ldns[-1][self.action_dim:], synapse=0)
                    nengo.Connection(switch_node, ensembles[-1][self.action_dim:self.action_dim + self.state_dim],
                                     synapse=0.01)

                # if wanted, have the model predict the difference from the linear interpolation instead
                #if True:
                    #nengo.Connection(grad, z_preds[-1], transform=t_d)

                # apply the error to the learning rule
                nengo.Connection(errors[-1], self.connections[-1].learning_rule)

                prediction = {
                    f"{i}": {
                        "delay": t_d,
                        # record the prediction
                        "states_pred": nengo.Probe(z_preds[-1]),
                        # record the error
                        "errors": nengo.Probe(errors[-1]),
                    }
                }

                recordings["predictions"].update(prediction)

        return model, recordings


"""
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

"""